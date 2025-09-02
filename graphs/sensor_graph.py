
import time, asyncio, logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import init_metrics, node_runs_total, node_duration_seconds, cooling_global_temp_c, cabinet_temp_c, energy_kw, efficiency_score
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import set_latest_state, get_config
from common.otel import init_tracer
from common.logging import setup_json_logging
from env.frontier_env import SmallFrontierModel

setup_json_logging()
init_metrics()
tracer = init_tracer("sensor-graph")
log = logging.getLogger("sensor-graph")

# Properly integrated SmallFrontierModel: construct once, reuse
_env = SmallFrontierModel()  # assumes .step() -> dict of temps & optionally energy

def step_env(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="sensor_graph"; nname="step_env"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        temps = {}
        meta = {}
        try:
            obs = _env.step()
            if isinstance(obs, dict):  # expect keys like cabinet_1.., cooling_tower, energy_kw, etc.
                temps = {k: float(v) for k,v in obs.items() if "cabinet" in k or "cooling" in k}
                if "energy_kw" in obs: meta["energy_kw"] = float(obs["energy_kw"])
        except Exception as e:
            log.error(f"env.step failed: {e}")
            temps = {f"cabinet_{i}": 25.0 for i in range(1,6)}; temps["cooling_tower"]=22.0
        vals = [v for k,v in temps.items() if k.startswith("cabinet_")]
        if vals:
            avg=sum(vals)/len(vals); cooling_global_temp_c.set(avg)
            for i,v in enumerate(vals, start=1):
                cabinet_temp_c.labels(cabinet=f"C{i}").set(v)
            # naive efficiency proxy
            if meta.get("energy_kw") is not None and avg:
                energy_kw.set(meta["energy_kw"])
                efficiency_score.set(max(0.0, min(1.0, 1.0 - (avg-24.0)/10.0)))
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return {"temps": temps, "meta": meta, "ts": time.time()}

def publish_state(state: Dict[str, Any]) -> Dict[str, Any]:
    gname="sensor_graph"; nname="publish_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    with tracer.start_as_current_span(nname):
        payload = {"type":"telemetry.state","data":state}
        asyncio.run(nats_publish(ROUTING["state_out"], payload, agent="sensor"))
        set_latest_state(state)
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

builder = StateGraph(dict)
builder.add_node("step_env", step_env)
builder.add_node("publish_state", publish_state)
builder.add_edge("step_env","publish_state")
builder.add_edge("publish_state", END)
builder.set_entry_point("step_env")
graph = builder.compile()
