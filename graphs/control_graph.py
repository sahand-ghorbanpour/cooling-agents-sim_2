
import time, asyncio, os, requests, logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import init_metrics, node_runs_total, node_duration_seconds, control_action, llm_request_duration_seconds
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import get_latest_state
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("control-graph")
log = logging.getLogger("control-graph")

TARGET_TEMP = float(os.getenv("TARGET_TEMP_C", "24.0"))

def get_state(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="control_graph"; nname="get_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        if "temps" not in state:
            latest = get_latest_state({})
            state.update(latest if latest else {})
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def guardrails_validate(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="control_graph"; nname="guardrails_validate"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        try:
            r = requests.post("http://guardrails:8001/validate", json={"content": {"temps": state.get("temps", {})}}, timeout=2.0)
            jr = r.json()
            state["guardrails_ok"] = jr.get("ok", True)
        except Exception as e:
            log.warning(f"guardrails error: {e}")
            state["guardrails_ok"] = True
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def propose_targets(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="control_graph"; nname="propose_targets"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        if not state.get("guardrails_ok", True):
            state["proposed"] = {}
            return state
        temps = state.get("temps", {})
        model = os.getenv("NIM_MODEL","meta/llama-3.1-8b-instruct")
        try:
            t1=time.time()
            resp = requests.post("http://llm-gateway:9000/plan", json={"temps": temps, "target": TARGET_TEMP}, timeout=10)
            llm_request_duration_seconds.labels(model).observe(time.time()-t1)
            if resp.ok:
                plan = resp.json()
                state["proposed"] = plan.get("actions", {})
        except Exception as e:
            log.warning(f"llm-gateway failed: {e}")
        # fallback P-control
        if "proposed" not in state:
            actions = {}
            for k,v in temps.items():
                if k.startswith("cabinet_"):
                    error = float(v) - TARGET_TEMP
                    pct = max(0.0, min(100.0, error*15.0 + 10.0))
                    actions[k+"_fan_pct"] = round(pct, 1)
            actions["cooling_tower_speed_pct"] = 50.0
            state["proposed"] = actions
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def clamp_and_enforce(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="control_graph"; nname="clamp_and_enforce"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        actions = state.get("proposed", {})
        clamped = {}
        for key,val in actions.items():
            clamped[key] = max(0.0, min(100.0, float(val)))
            control_action.labels(target=key).set(clamped[key])
        state["actions"] = clamped
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def publish_actions(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="control_graph"; nname="publish_actions"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        payload = {"type":"control.actions","data":state.get("actions", {})}
        asyncio.run(nats_publish(ROUTING["actions_out"], payload, agent="control"))
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

builder = StateGraph(dict)
builder.add_node("get_state", get_state)
builder.add_node("guardrails_validate", guardrails_validate)
builder.add_node("propose_targets", propose_targets)
builder.add_node("clamp_and_enforce", clamp_and_enforce)
builder.add_node("publish_actions", publish_actions)
builder.add_edge("get_state","guardrails_validate")
builder.add_edge("guardrails_validate","propose_targets")
builder.add_edge("propose_targets","clamp_and_enforce")
builder.add_edge("clamp_and_enforce","publish_actions")
builder.add_edge("publish_actions", END)
builder.set_entry_point("get_state")
graph = builder.compile()
