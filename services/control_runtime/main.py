# services/control_runtime/main.py - LLM-Only Control with JSON Schema
import asyncio, json, os, logging, time, requests
from typing import Dict, Any, Optional
import numpy as np
from nats.aio.client import Client as NATS
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from common.logging import setup_json_logging
from common.config import get_config, set_config
from common.nats_utils import get_nats_connection, publish as nats_publish

log = setup_json_logging()
start_http_server(int(os.getenv("METRICS_PORT","9010")))

# Prometheus metrics
control_loops_total = Counter('control_loops_total', 'Total control loop executions')
control_loop_duration = Histogram('control_loop_duration_seconds', 'Control loop execution time')
control_action_applied = Gauge('control_action_applied', 'Applied control action', ['target', 'action_type'])
llm_requests_total = Counter('llm_requests_total', 'Total LLM requests', ['status'])
llm_retry_attempts = Counter('llm_retry_attempts_total', 'LLM retry attempts')
control_error = Gauge('control_error_celsius', 'Control error from target temperature')

# JSON Schema for LLM responses
CONTROL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "object",
            "properties": {
                "cabinet_1": {
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
                    "required": ["secondary_temp_adjustment", "pressure_differential", "valve_positions"]
                },
                "cabinet_2": {
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
                    "required": ["secondary_temp_adjustment", "pressure_differential", "valve_positions"]
                },
                "cabinet_3": {
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
                    "required": ["secondary_temp_adjustment", "pressure_differential", "valve_positions"]
                },
                "cabinet_4": {
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
                    "required": ["secondary_temp_adjustment", "pressure_differential", "valve_positions"]
                },
                "cabinet_5": {
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
                    "required": ["secondary_temp_adjustment", "pressure_differential", "valve_positions"]
                },
                "cooling_tower": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "integer", "minimum": 0, "maximum": 8}
                    },
                    "required": ["action"]
                }
            },
            "required": ["cabinet_1", "cabinet_2", "cabinet_3", "cabinet_4", "cabinet_5", "cooling_tower"]
        },
        "reasoning": {
            "type": "string",
            "minLength": 10,
            "maxLength": 500
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "emergency": {
            "type": "boolean"
        }
    },
    "required": ["actions", "reasoning", "confidence", "emergency"]
}

class LLMControlState:
    def __init__(self):
        self.target_temp_c = float(os.getenv("TARGET_TEMP_C", "24.0"))
        self.current_temps = {}
        self.last_actions = {}
        self.control_history = []
        self.last_time = time.time()
        self.llm_gateway_url = os.getenv("LLM_GATEWAY_URL", "http://llm-gateway:9000")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.fallback_enabled = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"
        
state = LLMControlState()

async def main():
    """Main control runtime loop"""
    nc = await get_nats_connection()
    
    # Subscribe to telemetry updates
    await nc.subscribe("dc.telemetry.state", cb=handle_telemetry)
    
    # Subscribe to coordination messages
    await nc.subscribe("dc.control.coordinate", cb=handle_coordination)
    
    log.info("LLM Control runtime started - subscriptions active")
    
    # Main control loop
    while True:
        try:
            await run_llm_control_loop()
            
            # Control frequency
            loop_interval = float(get_config("control_loop_interval_ms", 2000)) / 1000.0  # Slower for LLM
            await asyncio.sleep(loop_interval)
            
        except Exception as e:
            log.error(f"Control loop error: {e}")
            await asyncio.sleep(2.0)

async def handle_telemetry(msg):
    """Handle incoming sensor telemetry"""
    try:
        payload = json.loads(msg.data.decode())
        # Handle both old nested format and new direct format
        if "data" in payload:
            temps = payload["data"].get("temps", {})  # Old format
        else:
            temps = payload.get("temps", {})          # New format ✅
        
        if temps:
            state.current_temps.update(temps)
            log.debug(f"Updated temperatures: {len(temps)} sensors")
            
    except Exception as e:
        log.error(f"Error handling telemetry: {e}")

async def handle_coordination(msg):
    """Handle coordination messages"""
    try:
        payload = json.loads(msg.data.decode())
        command = payload.get("command")
        
        if command == "update_target":
            new_target = float(payload.get("target_temp_c", state.target_temp_c))
            state.target_temp_c = new_target
            log.info(f"Updated target temperature to {new_target}°C")
            
        elif command == "emergency_mode":
            # Force immediate LLM control decision
            await run_llm_control_loop(emergency=True)
            
    except Exception as e:
        log.error(f"Error handling coordination: {e}")

async def run_llm_control_loop(emergency: bool = False):
    """Execute LLM-based control loop"""
    with control_loop_duration.time():
        control_loops_total.inc()
        current_time = time.time()
        
        if not state.current_temps:
            log.warning("No temperature data available for control")
            return
        
        # Calculate system state
        cabinet_temps = {k: v for k, v in state.current_temps.items() if k.startswith("cabinet_")}
        if not cabinet_temps:
            return
        
        global_temp = sum(cabinet_temps.values()) / len(cabinet_temps)
        global_error = global_temp - state.target_temp_c
        control_error.set(global_error)
        
        # Request LLM control decision
        control_decision = await request_llm_control_decision(
            temps=state.current_temps,
            target_temp=state.target_temp_c,
            global_error=global_error,
            emergency=emergency or abs(global_error) > 3.0
        )
        
        if control_decision:
            # Convert to simulation format and publish
            sim_actions = convert_llm_to_simulation_actions(control_decision["actions"])
            
            # Publish to simulation
            sim_payload = {
                "type": "control.actions",
                "data": sim_actions,
                "timestamp": current_time,
                "source": "llm_control_runtime",
                "reasoning": control_decision.get("reasoning", ""),
                "confidence": control_decision.get("confidence", 0.5)
            }
            
            await nats_publish("dc.control.actions", sim_payload, agent="llm_control")
            
            # Update metrics
            for i in range(1, 6):
                cabinet_key = f"cabinet_{i}"
                if cabinet_key in control_decision["actions"]:
                    action_data = control_decision["actions"][cabinet_key]
                    temp_adj = action_data["secondary_temp_adjustment"]
                    control_action_applied.labels(target=f"cabinet_{i}", action_type="temp_adj").set(temp_adj)
            
            ct_action = control_decision["actions"]["cooling_tower"]["action"]
            control_action_applied.labels(target="cooling_tower", action_type="discrete").set(ct_action)
            
            # Store for history
            state.last_actions = control_decision["actions"]
            state.control_history.append({
                "timestamp": current_time,
                "actions": control_decision["actions"],
                "reasoning": control_decision.get("reasoning", ""),
                "confidence": control_decision.get("confidence", 0.5),
                "global_error": global_error
            })
            
            # Trim history
            if len(state.control_history) > 50:
                state.control_history = state.control_history[-50:]
                
            log.info(f"LLM control decision applied - confidence: {control_decision.get('confidence', 0.5):.2f}, "
                    f"error: {global_error:.2f}°C")

async def request_llm_control_decision(temps: Dict[str, float], target_temp: float, 
                                     global_error: float, emergency: bool = False) -> Optional[Dict[str, Any]]:
    """Request control decision from LLM with JSON schema validation"""
    
    # Prepare context for LLM
    context = {
        "current_temperatures": temps,
        "target_temperature_c": target_temp,
        "temperature_error": global_error,
        "emergency_mode": emergency,
        "historical_actions": state.control_history[-5:] if state.control_history else [],
        "system_constraints": {
            "min_temp": 20.0,
            "max_temp": 35.0,
            "max_cooling_tower_action": 8,
            "valve_position_sum_must_equal_1": True
        }
    }
    
    # System prompt with schema instructions
    system_prompt = f"""You are an expert datacenter cooling control system. Your job is to control 5 server cabinets and a cooling tower to maintain target temperature.

CRITICAL: You MUST respond with valid JSON matching this exact schema:
{json.dumps(CONTROL_RESPONSE_SCHEMA, indent=2)}

Control Parameters:
- secondary_temp_adjustment: [-1, 1] where -1 = more cooling, +1 = less cooling
- pressure_differential: [-1, 1] where higher = more flow
- valve_positions: [primary, secondary, bypass] must sum to 1.0
- cooling_tower action: 0-8 where higher = more cooling

Current situation: Target {target_temp}°C, Error {global_error:.2f}°C, Emergency: {emergency}

Respond ONLY with valid JSON. No other text."""

    user_prompt = f"Control decision needed:\n{json.dumps(context, indent=2)}"
    
    # Try LLM request with retries
    for attempt in range(state.max_retries):
        try:
            llm_retry_attempts.inc()
            
            # Request with schema guidance (NIM/OpenAI compatible)
            response = await make_llm_request(system_prompt, user_prompt, attempt)
            
            if response:
                # Validate response against schema
                validated_response = validate_llm_response(response)
                if validated_response:
                    llm_requests_total.labels(status="success").inc()
                    return validated_response
                else:
                    log.warning(f"LLM response validation failed (attempt {attempt + 1})")
                    
        except Exception as e:
            log.error(f"LLM request failed (attempt {attempt + 1}): {e}")
        
        # Wait before retry
        if attempt < state.max_retries - 1:
            await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    # All retries failed
    llm_requests_total.labels(status="failed").inc()
    
    if state.fallback_enabled:
        log.warning("LLM control failed - using fallback logic")
        return generate_fallback_control_decision(temps, target_temp, global_error, emergency)
    else:
        log.error("LLM control failed and fallback disabled")
        return None

async def make_llm_request(system_prompt: str, user_prompt: str, attempt: int) -> Optional[Dict[str, Any]]:
    """Make LLM request with timeout"""
    try:
        # For NIM/OpenAI compatible request
        payload = {
            "model": os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,  # Low temperature for consistent control
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}  # JSON mode if supported
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('NGC_API_KEY', 'dummy')}",
            "Content-Type": "application/json"
        }
        
        # Use async HTTP client
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=60.0)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{state.llm_gateway_url}/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Parse JSON response
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # Try to extract JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            return json.loads(json_match.group(0))
                        return None
                else:
                    log.error(f"LLM request failed with status {response.status}")
                    return None
                    
    except Exception as e:
        log.error(f"LLM HTTP request error: {e}")
        return None

def validate_llm_response(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate LLM response against schema"""
    try:
        # Basic structure validation
        if "actions" not in response:
            return None
        
        actions = response["actions"]
        
        # Validate cabinet actions
        for i in range(1, 6):
            cabinet_key = f"cabinet_{i}"
            if cabinet_key not in actions:
                return None
            
            cabinet_action = actions[cabinet_key]
            
            # Check required fields
            required_fields = ["secondary_temp_adjustment", "pressure_differential", "valve_positions"]
            if not all(field in cabinet_action for field in required_fields):
                return None
            
            # Validate ranges
            temp_adj = cabinet_action["secondary_temp_adjustment"]
            if not (-1.0 <= temp_adj <= 1.0):
                return None
            
            pressure = cabinet_action["pressure_differential"]
            if not (-1.0 <= pressure <= 1.0):
                return None
            
            valve_positions = cabinet_action["valve_positions"]
            if (len(valve_positions) != 3 or 
                not all(0.0 <= v <= 1.0 for v in valve_positions) or
                abs(sum(valve_positions) - 1.0) > 0.01):  # Small tolerance
                # Normalize valve positions if close
                if abs(sum(valve_positions) - 1.0) <= 0.1:
                    total = sum(valve_positions)
                    cabinet_action["valve_positions"] = [v / total for v in valve_positions]
                else:
                    return None
        
        # Validate cooling tower
        if "cooling_tower" not in actions:
            return None
        
        ct_action = actions["cooling_tower"]["action"]
        if not (0 <= ct_action <= 8):
            return None
        
        # Validate other fields
        if "reasoning" not in response or len(response["reasoning"]) < 10:
            response["reasoning"] = "LLM control decision applied"
        
        if "confidence" not in response:
            response["confidence"] = 0.7
        
        if "emergency" not in response:
            response["emergency"] = False
        
        return response
        
    except Exception as e:
        log.error(f"Response validation error: {e}")
        return None

def generate_fallback_control_decision(temps: Dict[str, float], target_temp: float, 
                                     global_error: float, emergency: bool) -> Dict[str, Any]:
    """Generate fallback control decision when LLM fails"""
    
    fallback_actions = {
        "actions": {},
        "reasoning": "Fallback control logic - LLM unavailable",
        "confidence": 0.3,
        "emergency": emergency
    }
    
    # Simple proportional control for cabinets
    for i in range(1, 6):
        cabinet_key = f"cabinet_{i}"
        temp = temps.get(cabinet_key, target_temp)
        error = temp - target_temp
        
        # Simple control logic
        if emergency:
            # Emergency cooling
            temp_adj = -0.8 if error > 0 else 0.2
            pressure = 0.5 if error > 0 else -0.2
            valve_positions = [0.7, 0.2, 0.1] if error > 0 else [0.3, 0.4, 0.3]
        else:
            # Normal proportional control
            temp_adj = np.clip(-error * 0.3, -1.0, 1.0)
            pressure = np.clip(error * 0.2, -1.0, 1.0)
            
            # Balanced valve distribution
            cooling_need = np.clip((error + 2.0) / 4.0, 0.1, 0.9)
            valve_positions = [
                cooling_need * 0.5 + 0.2,
                cooling_need * 0.3 + 0.3,
                1.0 - (cooling_need * 0.8 + 0.5)
            ]
            # Normalize
            total = sum(valve_positions)
            valve_positions = [v / total for v in valve_positions]
        
        fallback_actions["actions"][cabinet_key] = {
            "secondary_temp_adjustment": float(temp_adj),
            "pressure_differential": float(pressure),
            "valve_positions": valve_positions
        }
    
    # Cooling tower action
    if emergency:
        ct_action = 8 if global_error > 0 else 2
    else:
        ct_action = max(0, min(8, int(4 + global_error * 2)))
    
    fallback_actions["actions"]["cooling_tower"] = {"action": ct_action}
    
    return fallback_actions

def convert_llm_to_simulation_actions(llm_actions: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LLM actions to simulation format - JSON safe"""
    sim_actions = {}
    
    # Convert cabinet actions
    for i in range(1, 6):
        cabinet_key = f"cabinet_{i}"
        sim_key = f"cdu-cabinet-{i}"
        
        if cabinet_key in llm_actions:
            action_data = llm_actions[cabinet_key]
            
            # Return as plain Python list (JSON serializable)
            sim_actions[sim_key] = [
                float(action_data["secondary_temp_adjustment"]),
                float(action_data["pressure_differential"]),
                float(action_data["valve_positions"][0]),
                float(action_data["valve_positions"][1]),
                float(action_data["valve_positions"][2])
            ]
    
    # Cooling tower action 
    if "cooling_tower" in llm_actions:
        sim_actions["cooling-tower-1"] = int(llm_actions["cooling_tower"]["action"])
    
    return sim_actions

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("LLM Control runtime stopped")
    except Exception as e:
        log.error(f"LLM Control runtime startup failed: {e}")
        exit(1)