
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import os, json, time, re, requests, logging
from prometheus_client import start_http_server, Histogram
from common.otel import init_tracer
from common.logging import setup_json_logging

log = setup_json_logging()
app = FastAPI(title="LLM Gateway (NIM/OpenAI proxy)")
init_tracer("llm-gateway")
start_http_server(int(os.getenv("METRICS_PORT","9002")))
LLM_BASE = os.getenv("OPENAI_BASE_URL", os.getenv("NIM_URL","http://nim-llm:8000")) + "/v1"
LLM_MODEL = os.getenv("NIM_MODEL","meta/llama-3.1-8b-instruct")
API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("NGC_API_KEY","nokey"))
HIST = Histogram("llm_gateway_request_seconds","Latency to upstream LLM")

class PlanReq(BaseModel):
    temps: Dict[str,float]
    target: float

class EmbReq(BaseModel):
    input: List[str]
    model: str | None = None

@app.post("/plan")
def plan(req: PlanReq):
    t0=time.time()
    try:
        headers={"Authorization": f"Bearer {API_KEY}"}
        url=f"{LLM_BASE}/chat/completions"
        sys_prompt=("You are a cooling control planner. Return STRICT JSON: "
                    "{'actions': {'cabinet_1_fan_pct': float, ..., 'cooling_tower_speed_pct': float}} "
                    "No prose.")
        user=f"temps={json.dumps(req.temps)}, target_c={req.target}"
        payload={"model": LLM_MODEL, "messages":[{"role":"system","content":sys_prompt},{"role":"user","content":user}], "temperature":0.2}
        r=requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data=r.json()
        content=data["choices"][0]["message"]["content"]
        actions={}
        try:
            parsed=json.loads(content); actions=parsed.get("actions", parsed)
        except Exception:
            m=re.search(r"\{[\s\S]*\}", content)
            if m:
                try:
                    tmp=json.loads(m.group(0)); actions=tmp.get("actions", tmp)
                except Exception: actions={}
        if not actions:
            avg=sum(req.temps.values())/max(len(req.temps),1)
            actions={f"cabinet_{i}_fan_pct": max(10.0, min(100.0,(req.temps.get(f'cabinet_{i}',avg)-req.target)*12+30)) for i in range(1,6)}
            actions["cooling_tower_speed_pct"]=max(10.0,min(100.0,(avg-req.target)*10+50))
        return {"actions": actions, "model": LLM_MODEL, "upstream": LLM_BASE}
    finally:
        HIST.observe(time.time()-t0)

@app.post("/embeddings")
def embeddings(req: EmbReq):
    headers={"Authorization": f"Bearer {API_KEY}"}
    url=f"{LLM_BASE}/embeddings"
    payload={"model": req.model or os.getenv("EMBED_MODEL","nvidia/nv-embed-v1"), "input": req.input}
    r=requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()
