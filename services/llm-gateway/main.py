# services/llm-gateway/main.py
# FastAPI gateway for NIM LLM (OpenAI-compatible)
# - Preserves existing endpoints and metrics
# - Adds robust schema handling via NIM nvext.guided_json
# - Uses safe response_format and timeouts to avoid stalls

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import os, json, time, re, requests, logging
from prometheus_client import start_http_server, Histogram, Counter
from common.otel import init_tracer
from common.logging import setup_json_logging

# ----------------------------
# Logging / App / Metrics
# ----------------------------
log = setup_json_logging()
app = FastAPI(title="LLM Gateway with JSON Schema Support")
init_tracer("llm-gateway")
start_http_server(int(os.getenv("METRICS_PORT", "9002")))

LLM_BASE = os.getenv("OPENAI_BASE_URL", os.getenv("NIM_URL", "http://nim-llm:8000")) + "/v1"
LLM_MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("NGC_API_KEY", "nokey"))

CONNECT_TIMEOUT = float(os.getenv("CONNECT_TIMEOUT_SEC", 5))
READ_TIMEOUT = float(os.getenv("READ_TIMEOUT_SEC", 120))

# Metrics
llm_requests_total = Counter("llm_requests_total", "Total LLM requests", ["endpoint", "status"])
llm_request_duration = Histogram("llm_request_duration_seconds", "LLM request latency", ["endpoint"])
json_validation_errors = Counter("json_validation_errors_total", "JSON validation failures")

# ----------------------------
# Models
# ----------------------------
class ControlRequest(BaseModel):
    temps: Dict[str, float] = Field(..., description="Current temperature readings")
    target: float = Field(..., description="Target temperature in Celsius")
    emergency: bool = Field(default=False, description="Emergency mode flag")
    history: Optional[List[Dict]] = Field(default=None, description="Control history")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="System constraints")

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    response_format: Optional[Dict[str, Any]] = None  # expects {"type": "text"|"json_object"}
    schema: Optional[Dict[str, Any]] = None           # OpenAI-style top-level schema (we map to nvext)

class EmbReq(BaseModel):
    input: List[str]
    model: Optional[str] = None

# ----------------------------
# Control schema (fixed)
# ----------------------------
CONTROL_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "object",
            "properties": {
                **{
                    f"cabinet_{i}": {
                        "type": "object",
                        "properties": {
                            "secondary_temp_adjustment": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                            "pressure_differential": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                            "valve_positions": {
                                "type": "array",
                                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "minItems": 3,
                                "maxItems": 3
                            }
                        },
                        "required": ["secondary_temp_adjustment", "pressure_differential", "valve_positions"],
                        "additionalProperties": False
                    }
                    for i in range(1, 6)
                },
                "cooling_tower": {
                    "type": "object",
                    "properties": {"action": {"type": "integer", "minimum": 0, "maximum": 8}},
                    "required": ["action"],
                    "additionalProperties": False
                }
            },
            "required": [*(f"cabinet_{i}" for i in range(1, 6)), "cooling_tower"],
            "additionalProperties": False
        },
        "reasoning": {"type": "string", "minLength": 10, "maxLength": 500},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "emergency": {"type": "boolean"}
    },
    "required": ["actions", "reasoning", "confidence", "emergency"],
    "additionalProperties": False
}

# ----------------------------
# Utilities
# ----------------------------
def _truncate(obj: Any, limit: int = 800) -> str:
    try:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return (s[: limit - 3] + "...") if len(s) > limit else s

def _post_upstream(path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{LLM_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
    log.debug("POST %s payload=%s", url, _truncate(payload, 1000))
    return requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))

def _normalize_chat_payload(req: ChatCompletionRequest) -> Dict[str, Any]:
    """Turn client OpenAI-style request into NIM-friendly payload:
       - Only 'text' or 'json_object' response_format
       - If 'schema' present, map to nvext.guided_json
    """
    payload: Dict[str, Any] = {
        "model": req.model or LLM_MODEL,
        "messages": req.messages,
        "temperature": 0.0 if req.temperature is None else req.temperature,
        "max_tokens": min(int(req.max_tokens or 512), 2048),
    }

    # response_format allowed by NIM: "text" or "json_object"
    rf = req.response_format or {"type": "json_object"}
    if isinstance(rf, dict) and rf.get("type") in ("text", "json_object"):
        payload["response_format"] = rf
    else:
        payload["response_format"] = {"type": "json_object"}

    # If client provided a schema, map to NIM's nvext guided_json
    if req.schema:
        payload["nvext"] = {"type": "guided_json", "schema": req.schema}
        # Nudge to JSON-only
        msgs = payload["messages"] or []
        if msgs and msgs[0].get("role") == "system":
            msgs[0]["content"] = (msgs[0]["content"] + "\nRespond ONLY with valid JSON.").strip()
        else:
            msgs.insert(0, {"role": "system", "content": "Respond ONLY with valid JSON."})
        payload["messages"] = msgs
        log.info("Applying guided_json schema via nvext: %s", _truncate(req.schema, 600))

    return payload

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/control_decision")
def control_decision(req: ControlRequest):
    """Generate structured control decision with schema validation"""
    endpoint = "control_decision"
    t0 = time.time()
    try:
        llm_requests_total.labels(endpoint=endpoint, status="started").inc()

        system_prompt = create_control_system_prompt(req.emergency)
        user_prompt = create_control_user_prompt(req)

        response = make_structured_llm_request(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=CONTROL_SCHEMA,
            temperature=0.1
        )

        if response:
            validated = validate_control_response(response, req)
            llm_requests_total.labels(endpoint=endpoint, status="success").inc()
            return {
                "decision": validated,
                "model": LLM_MODEL,
                "processing_time_ms": (time.time() - t0) * 1000,
                "source": "structured_llm"
            }
        else:
            raise HTTPException(status_code=500, detail="LLM request failed")

    except Exception as e:
        llm_requests_total.labels(endpoint=endpoint, status="error").inc()
        log.error(f"Control decision error: {e}")
        fallback = generate_fallback_control(req)
        return {
            "decision": fallback,
            "model": "fallback",
            "processing_time_ms": (time.time() - t0) * 1000,
            "source": "fallback_logic",
            "error": str(e)
        }
    finally:
        llm_request_duration.labels(endpoint=endpoint).observe(time.time() - t0)

@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions with optional schema support (nvext guided_json)"""
    endpoint = "chat_completions"
    t0 = time.time()
    try:
        llm_requests_total.labels(endpoint=endpoint, status="started").inc()

        payload = _normalize_chat_payload(req)
        r = _post_upstream("/chat/completions", payload)

        if not r.ok:
            # bubble upstream error cleanly
            try:
                err = r.json()
            except Exception:
                err = {"message": r.text}
            llm_requests_total.labels(endpoint=endpoint, status="upstream_error").inc()
            return err, r.status_code

        resp = r.json()

        # Optional quick validation if schema present (best-effort)
        if req.schema:
            try:
                content = resp["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if not validate_against_schema(parsed, req.schema):
                    json_validation_errors.inc()
                    log.warning("Upstream response did not match provided schema")
            except Exception:
                json_validation_errors.inc()
                log.warning("Non-JSON response despite schema requirement")

        llm_requests_total.labels(endpoint=endpoint, status="success").inc()
        return resp

    except Exception as e:
        llm_requests_total.labels(endpoint=endpoint, status="error").inc()
        log.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat completion error: {e}")
    finally:
        llm_request_duration.labels(endpoint=endpoint).observe(time.time() - t0)

@app.post("/plan")
def plan(req: ControlRequest):
    """Legacy endpoint for backward compatibility"""
    control_response = control_decision(req)
    decision = control_response["decision"]
    legacy_actions = {}

    # Convert cabinet actions to legacy fan percentages
    for i in range(1, 6):
        cabinet_key = f"cabinet_{i}"
        if cabinet_key in decision["actions"]:
            act = decision["actions"][cabinet_key]
            temp_adj = act["secondary_temp_adjustment"]
            fan_pct = max(10.0, min(100.0, 50.0 - temp_adj * 40.0))
            legacy_actions[f"cabinet_{i}_fan_pct"] = fan_pct

    # Convert cooling tower
    if "cooling_tower" in decision["actions"]:
        ct_action = decision["actions"]["cooling_tower"]["action"]
        ct_speed_pct = 10.0 + (ct_action / 8.0) * 80.0
        legacy_actions["cooling_tower_speed_pct"] = ct_speed_pct

    return {
        "actions": legacy_actions,
        "model": LLM_MODEL,
        "upstream": LLM_BASE,
        "reasoning": decision.get("reasoning", ""),
        "confidence": decision.get("confidence", 0.5)
    }

@app.post("/embeddings")
def embeddings(req: EmbReq):
    """Embeddings passthrough"""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{LLM_BASE}/embeddings"
    payload = {"model": req.model or os.getenv("EMBED_MODEL", "nvidia/nv-embed-v1"), "input": req.input}
    r = requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()
    return r.json()

# ----------------------------
# Control prompts & helpers
# ----------------------------
def create_control_system_prompt(emergency: bool) -> str:
    return f"""You are an expert datacenter cooling control AI. Control 5 server cabinets and 1 cooling tower to maintain optimal temperatures.

CRITICAL: Respond with VALID JSON matching the exact schema provided.

Control Parameters:
- secondary_temp_adjustment: [-1, 1] where -1 = maximum cooling, +1 = minimum cooling
- pressure_differential: [-1, 1] where higher = more flow pressure
- valve_positions: [primary, secondary, bypass] - MUST sum to 1.0
- cooling_tower action: integer 0-8 where higher = more cooling

Emergency Mode: {"ACTIVE - prioritize immediate cooling" if emergency else "Normal operation"}

Guidelines:
- If temperature > target: use negative temp_adjustment, higher cooling_tower action
- If temperature < target: use positive temp_adjustment, lower cooling_tower action
- Emergency: maximize cooling (temp_adj near -1, cooling_tower 7-8)
- Valve positions: [0.6, 0.3, 0.1] for cooling, [0.3, 0.4, 0.3] for normal, [0.2, 0.2, 0.6] for minimal
- Provide clear reasoning for your decisions

Respond ONLY with valid JSON. No other text."""

def create_control_user_prompt(req: ControlRequest) -> str:
    cabinet_temps = {k: v for k, v in req.temps.items() if k.startswith("cabinet_")}
    avg_temp = sum(cabinet_temps.values()) / len(cabinet_temps) if cabinet_temps else req.target
    error = avg_temp - req.target

    context = {
        "temperatures": req.temps,
        "target_temperature": req.target,
        "average_temperature": round(avg_temp, 2),
        "temperature_error": round(error, 2),
        "emergency_mode": req.emergency,
        "severity": "CRITICAL" if abs(error) > 3 else "HIGH" if abs(error) > 1.5 else "NORMAL"
    }

    if req.history:
        context["recent_actions"] = req.history[-3:] if len(req.history) > 3 else req.history

    return f"Current system state:\n{json.dumps(context, indent=2)}\n\nGenerate control decision:"

def make_structured_llm_request(
    system_prompt: str,
    user_prompt: str,
    schema: Dict[str, Any],
    temperature: float = 0.1
) -> Optional[Dict[str, Any]]:
    """Structured call using NIM nvext guided_json (no giant schema pasted into prompt)."""
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json", "Accept": "application/json"}
        url = f"{LLM_BASE}/chat/completions"

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt + "\nRespond ONLY with valid JSON."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 768,
            "response_format": {"type": "json_object"},
            "nvext": {"type": "guided_json", "schema": schema}
        }

        r = requests.post(url, headers=headers, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]

        # Parse JSON response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return None

    except Exception as e:
        log.error(f"Structured LLM request failed: {e}")
        return None

def validate_control_response(response: Dict[str, Any], req: ControlRequest) -> Dict[str, Any]:
    """Validate and fix control response"""
    response.setdefault("actions", {})

    # Ensure all required cabinet actions exist
    for i in range(1, 6):
        cabinet_key = f"cabinet_{i}"
        if cabinet_key not in response["actions"]:
            temp = req.temps.get(cabinet_key, req.target)
            error = temp - req.target
            response["actions"][cabinet_key] = {
                "secondary_temp_adjustment": max(-1.0, min(1.0, -error * 0.3)),
                "pressure_differential": max(-1.0, min(1.0, error * 0.2)),
                "valve_positions": [0.4, 0.3, 0.3]
            }

        # Normalize valve positions
        valve_positions = response["actions"][cabinet_key].get("valve_positions", [0.4, 0.3, 0.3])
        if len(valve_positions) == 3:
            total = sum(valve_positions) or 1.0
            response["actions"][cabinet_key]["valve_positions"] = [v / total for v in valve_positions]

    # Ensure cooling tower action exists
    if "cooling_tower" not in response["actions"]:
        avg_temp = sum(req.temps.values()) / len(req.temps) if req.temps else req.target
        error = avg_temp - req.target
        ct_action = max(0, min(8, int(4 + error * 2)))
        response["actions"]["cooling_tower"] = {"action": ct_action}

    # Set defaults for missing fields
    response.setdefault("reasoning", "Automated control decision")
    response.setdefault("confidence", 0.7)
    response.setdefault("emergency", req.emergency)

    return response

def validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Basic schema validation (best-effort). For production, use jsonschema."""
    try:
        if schema.get("type") == "object":
            required = schema.get("required", [])
            for field in required:
                if field not in data:
                    return False
        return True
    except Exception:
        return False

def generate_fallback_control(req: ControlRequest) -> Dict[str, Any]:
    """Fallback proportional control if LLM fails"""
    actions: Dict[str, Any] = {}
    for i in range(1, 6):
        cabinet_key = f"cabinet_{i}"
        temp = req.temps.get(cabinet_key, req.target)
        error = temp - req.target

        if req.emergency:
            temp_adj = -0.9 if error > 0 else 0.3
            pressure = 0.6 if error > 0 else -0.3
            valve_positions = [0.7, 0.2, 0.1] if error > 0 else [0.2, 0.3, 0.5]
        else:
            temp_adj = max(-1.0, min(1.0, -error * 0.4))
            pressure = max(-1.0, min(1.0, error * 0.3))
            cooling_factor = max(0.1, min(0.9, (error + 2.0) / 4.0))
            valve_positions = [
                0.2 + cooling_factor * 0.5,
                0.3 + cooling_factor * 0.2,
                0.5 - cooling_factor * 0.7
            ]
            total = sum(valve_positions) or 1.0
            valve_positions = [v / total for v in valve_positions]

        actions[cabinet_key] = {
            "secondary_temp_adjustment": temp_adj,
            "pressure_differential": pressure,
            "valve_positions": valve_positions
        }

    avg_temp = sum(req.temps.values()) / len(req.temps) if req.temps else req.target
    error = avg_temp - req.target
    if req.emergency:
        ct_action = 8 if error > 0 else 1
    else:
        ct_action = max(0, min(8, int(4 + error * 2.5)))

    actions["cooling_tower"] = {"action": ct_action}

    return {
        "actions": actions,
        "reasoning": "Fallback proportional control - LLM service unavailable",
        "confidence": 0.4,
        "emergency": req.emergency
    }
