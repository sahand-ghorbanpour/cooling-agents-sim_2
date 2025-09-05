# graphs/sensor_graph.py - Optimized for direct data flow

import time, asyncio, json, logging, os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import (init_metrics, node_runs_total, node_duration_seconds, 
                          cooling_global_temp_c, cabinet_temp_c, energy_kw, efficiency_score)
from common.nats_utils import get_nats_connection
from common.config import get_latest_state, get_config
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("sensor-graph")
log = logging.getLogger("sensor-graph")

async def validate_and_process_telemetry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate incoming telemetry and process metrics - single optimized node"""
    gname="sensor_graph"; nname="validate_and_process_telemetry"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Get latest data directly from Redis (populated by sensor-agent)
        latest_data = get_latest_state({})
        
        if not latest_data:
            log.warning("No telemetry data available")
            state["status"] = "no_data"
            state["temps"] = {f"cabinet_{i}": 25.0 for i in range(1,6)}  # Fallback
            state["meta"] = {"energy_kw": 100.0, "source": "fallback"}
        else:
            state.update(latest_data)
            state["status"] = "healthy"
        
        # Process metrics directly in this node
        temps = state.get("temps", {})
        meta = state.get("meta", {})
        
        # Update Prometheus metrics
        cabinet_vals = [v for k,v in temps.items() if k.startswith("cabinet_")]
        if cabinet_vals:
            avg_temp = sum(cabinet_vals) / len(cabinet_vals)
            cooling_global_temp_c.set(avg_temp)
            
            for i, temp in enumerate(cabinet_vals, start=1):
                cabinet_temp_c.labels(cabinet=f"C{i}").set(temp)
            
            # Energy metrics
            energy = meta.get("energy_kw", 0)
            energy_kw.set(energy)
            
            # Efficiency calculation
            target_temp = float(get_config("target_temp_c", 24.0))
            temp_efficiency = max(0.0, min(1.0, 1.0 - abs(avg_temp - target_temp) / 10.0))
            energy_efficiency = max(0.0, min(1.0, 1.0 - energy / 200.0))
            overall_efficiency = (temp_efficiency + energy_efficiency) / 2.0
            efficiency_score.set(overall_efficiency)
        
        state["processed"] = True
        state["metrics_updated"] = True
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def health_check_and_alerting(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform health checks and generate alerts if needed"""
    gname="sensor_graph"; nname="health_check_and_alerting"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        temps = state.get("temps", {})
        alerts = []
        
        # Temperature range checks
        for cabinet_id, temp in temps.items():
            if cabinet_id.startswith("cabinet_"):
                if temp > 30.0:
                    alerts.append({
                        "type": "high_temperature",
                        "cabinet": cabinet_id,
                        "temperature": temp,
                        "severity": "critical" if temp > 35.0 else "warning"
                    })
                elif temp < 18.0:
                    alerts.append({
                        "type": "low_temperature", 
                        "cabinet": cabinet_id,
                        "temperature": temp,
                        "severity": "warning"
                    })
        
        # Energy efficiency checks
        energy = state.get("meta", {}).get("energy_kw", 0)
        if energy > 180.0:
            alerts.append({
                "type": "high_energy_consumption",
                "energy_kw": energy,
                "severity": "warning"
            })
        
        # Data freshness check
        ts = state.get("ts", time.time())
        data_age = time.time() - ts
        if data_age > 30.0:  # More than 30 seconds old
            alerts.append({
                "type": "stale_data",
                "age_seconds": data_age,
                "severity": "warning"
            })
        
        state["alerts"] = alerts
        state["health_status"] = "critical" if any(a["severity"] == "critical" for a in alerts) else \
                               "warning" if alerts else "healthy"
        
        # Publish alerts if any exist
        if alerts:
            nc = await get_nats_connection()
            alert_payload = {
                "type": "sensor.alerts",
                "alerts": alerts,
                "timestamp": time.time(),
                "source": "sensor_graph"
            }
            await nc.publish("dc.alerts", json.dumps(alert_payload).encode())
            log.warning(f"Published {len(alerts)} alerts")
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

# Simplified graph with only 2 nodes - eliminates 3 data hops
builder = StateGraph(dict)
builder.add_node("validate_and_process_telemetry", validate_and_process_telemetry)
builder.add_node("health_check_and_alerting", health_check_and_alerting)

# Simple linear flow
builder.add_edge("validate_and_process_telemetry", "health_check_and_alerting")
builder.add_edge("health_check_and_alerting", END)
builder.set_entry_point("validate_and_process_telemetry")

graph = builder.compile()

# Direct execution function for minimal overhead
async def run_optimized_sensor_processing():
    """Direct sensor processing without graph overhead for maximum performance"""
    try:
        # Get data directly
        latest_data = get_latest_state({})
        if not latest_data:
            return
        
        temps = latest_data.get("temps", {})
        meta = latest_data.get("meta", {})
        
        # Update metrics directly
        cabinet_vals = [v for k,v in temps.items() if k.startswith("cabinet_")]
        if cabinet_vals:
            avg_temp = sum(cabinet_vals) / len(cabinet_vals)
            cooling_global_temp_c.set(avg_temp)
            
            for i, temp in enumerate(cabinet_vals, start=1):
                cabinet_temp_c.labels(cabinet=f"C{i}").set(temp)
            
            energy = meta.get("energy_kw", 0)
            energy_kw.set(energy)
            
            # Quick efficiency calc
            target_temp = 24.0
            temp_eff = max(0.0, min(1.0, 1.0 - abs(avg_temp - target_temp) / 10.0))
            energy_eff = max(0.0, min(1.0, 1.0 - energy / 200.0))
            efficiency_score.set((temp_eff + energy_eff) / 2.0)
            
            log.debug(f"Metrics updated: avg_temp={avg_temp:.1f}°C, energy={energy:.1f}kW")
        
    except Exception as e:
        log.error(f"Optimized sensor processing error: {e}")

# Export both graph and direct function
__all__ = ["graph", "run_optimized_sensor_processing"]