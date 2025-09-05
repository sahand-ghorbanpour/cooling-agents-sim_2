
 #services/guardrails/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os, re, logging
from prometheus_client import start_http_server
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from common.otel import init_tracer
from common.logging import setup_json_logging

log = setup_json_logging()
app = FastAPI(title="Guardrails (policy)")
FastAPIInstrumentor.instrument_app(app)
init_tracer("guardrails")
start_http_server(int(os.getenv("METRICS_PORT","9003")))

# Simple policy: deny if temps missing or out-of-range; block unsafe high fan % later
class GuardReq(BaseModel):
    content: Dict[str,Any]

@app.post("/validate")
def validate(req: GuardReq):
    temps = req.content.get("temps", {})
    if not isinstance(temps, dict) or not temps:
        return {"ok": False, "issues": ["missing_temps"]}
    issues: List[str] = []
    for k,v in temps.items():
        try:
            fv=float(v)
            if fv<10 or fv>60: issues.append(f"temp_out_of_range:{k}")
        except Exception:
            issues.append(f"bad_number:{k}")
    ok = len(issues)==0
    return {"ok": ok, "issues": issues}
