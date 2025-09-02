
import time, asyncio, requests, logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import init_metrics, node_runs_total, node_duration_seconds
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import get_latest_state
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("orchestrator-graph")
log = logging.getLogger("orchestrator-graph")

def fetch_state(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="orchestrator_graph"; nname="fetch_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        state["latest"] = get_latest_state({})
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def optimize_targets(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="orchestrator_graph"; nname="optimize_targets"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        temps = (state.get("latest") or {}).get("temps", {})
        try:
            resp = requests.post("http://mcp-math:7000/optimize", json={"temps": temps}, timeout=2.0)
            if resp.ok:
                state["opt_targets"] = resp.json().get("targets", {})
        except Exception as e:
            log.warning(f"mcp-math call failed: {e}")
            state["opt_targets"] = {}
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def emit_control(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="orchestrator_graph"; nname="emit_control"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        asyncio.run(nats_publish(ROUTING["orch_targets"], {"targets": state.get("opt_targets", {})}, agent="orchestrator"))
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

builder = StateGraph(dict)
builder.add_node("fetch_state", fetch_state)
builder.add_node("optimize_targets", optimize_targets)
builder.add_node("emit_control", emit_control)
builder.add_edge("fetch_state","optimize_targets")
builder.add_edge("optimize_targets","emit_control")
builder.add_edge("emit_control", END)
builder.set_entry_point("fetch_state")
graph = builder.compile()
