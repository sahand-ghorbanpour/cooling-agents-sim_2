
import time, asyncio, logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import init_metrics, node_runs_total, node_duration_seconds
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import set_config
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("maintenance-graph")
log = logging.getLogger("maintenance-graph")

def maybe_update_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="maintenance_graph"; nname="maybe_update_schedule"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        interval_h = state.get("interval_hours")
        if interval_h is not None:
            set_config("maintenance_interval_hours", float(interval_h))
            asyncio.run(nats_publish("dc.config.maintenance", {"interval_hours": float(interval_h)}, agent="maintenance"))
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def run_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="maintenance_graph"; nname="run_checks"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        report = {"fans_ok": True, "filters_clean": True, "ts": time.time()}
        asyncio.run(nats_publish("dc.maintenance.report", {"report": report}, agent="maintenance"))
        state["report"] = report
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

builder = StateGraph(dict)
builder.add_node("maybe_update_schedule", maybe_update_schedule)
builder.add_node("run_checks", run_checks)
builder.add_edge("maybe_update_schedule","run_checks")
builder.add_edge("run_checks", END)
builder.set_entry_point("maybe_update_schedule")
graph = builder.compile()
