# graphs/sensor_graph.py - Modified for hybrid architecture
import time, asyncio, json, logging, os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import (init_metrics, node_runs_total, node_duration_seconds, 
                          cooling_global_temp_c, cabinet_temp_c, energy_kw, efficiency_score)
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import set_latest_state, get_config
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("sensor-graph")
log = logging.getLogger("sensor-graph")

# Global state to receive simulation data
_latest_simulation_data = None
_simulation_health = {"status": "unknown", "last_update": 0}

async def coordinate_sensors(state: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinate with external sensor-agent service"""
    gname="sensor_graph"; nname="coordinate_sensors"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Send trigger to external sensor-agent
        trigger_payload = {
            "type": "sensor.trigger",
            "timestamp": time.time(),
            "control_actions": state.get("control_actions"),  # Forward any control actions
            "source": "sensor_graph"
        }
        
        await nats_publish("simulation.trigger", trigger_payload, agent="sensor_coordinator")
        
        # Store coordination info
        state["coordination_sent"] = True
        state["trigger_timestamp"] = time.time()
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def wait_for_simulation(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wait for simulation results from external sensor-agent"""
    gname="sensor_graph"; nname="wait_for_simulation"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        global _latest_simulation_data, _simulation_health
        
        # Wait for fresh simulation data (with timeout)
        timeout = 10.0  # 10 second timeout
        start_wait = time.time()
        trigger_time = state.get("trigger_timestamp", 0)
        
        while time.time() - start_wait < timeout:
            if (_latest_simulation_data and 
                _latest_simulation_data.get("timestamp", 0) > trigger_time):
                # Got fresh data
                state["temps"] = _latest_simulation_data.get("temps", {})
                state["meta"] = _latest_simulation_data.get("meta", {})
                state["ts"] = _latest_simulation_data.get("timestamp", time.time())
                state["simulation_health"] = "healthy"
                break
                
            await asyncio.sleep(0.1)  # Check every 100ms
        
        else:
            # Timeout - use last known data or defaults
            log.warning("Simulation data timeout - using fallback")
            if _latest_simulation_data:
                state["temps"] = _latest_simulation_data.get("temps", {})
                state["meta"] = _latest_simulation_data.get("meta", {})
                state["simulation_health"] = "stale_data"
            else:
                # Complete fallback
                state["temps"] = {f"cabinet_{i}": 25.0 for i in range(1,6)}
                state["temps"]["cooling_tower"] = 22.0
                state["meta"] = {"energy_kw": 100.0, "error": "no_simulation_data"}
                state["simulation_health"] = "fallback"
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def process_telemetry(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process telemetry data and update metrics"""
    gname="sensor_graph"; nname="process_telemetry"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        temps = state.get("temps", {})
        meta = state.get("meta", {})
        
        # Update Prometheus metrics
        cabinet_vals = [v for k,v in temps.items() if k.startswith("cabinet_")]
        if cabinet_vals:
            avg_temp = sum(cabinet_vals) / len(cabinet_vals)
            cooling_global_temp_c.set(avg_temp)
            
            for i, temp in enumerate(cabinet_vals, start=1):
                cabinet_temp_c.labels(cabinet=f"C{i}").set(temp)
            
            # Update energy and efficiency metrics
            energy = meta.get("energy_kw", 0)
            energy_kw.set(energy)
            
            # Efficiency score
            temp_efficiency = max(0.0, min(1.0, 1.0 - abs(avg_temp - 24.0) / 10.0))
            energy_efficiency = max(0.0, min(1.0, 1.0 - energy / 200.0))
            overall_efficiency = (temp_efficiency + energy_efficiency) / 2.0
            efficiency_score.set(overall_efficiency)
        
        # Add processing metadata
        state["processed"] = True
        state["metrics_updated"] = True
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def publish_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Publish processed sensor state"""
    gname="sensor_graph"; nname="publish_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Prepare telemetry payload
        payload = {
            "type": "telemetry.state",
            "data": state,
            "timestamp": time.time(),
            "source": "sensor_graph_coordinator",
            "simulation_health": state.get("simulation_health", "unknown")
        }
        
        # Publish to multiple topics
        await nats_publish(ROUTING["state_out"], payload, agent="sensor")
        await nats_publish("simulation.state.processed", payload, agent="sensor")
        
        # Store latest state for other services
        set_latest_state(state)
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

# Function to receive simulation data from external sensor-agent
def update_simulation_data(data: Dict[str, Any]):
    """Called by NATS listener to update simulation data"""
    global _latest_simulation_data, _simulation_health
    _latest_simulation_data = data
    _simulation_health = {
        "status": "healthy",
        "last_update": time.time()
    }
    log.debug("Received simulation data update from sensor-agent")

# Build the sensor coordination graph
builder = StateGraph(dict)
builder.add_node("coordinate_sensors", coordinate_sensors)
builder.add_node("wait_for_simulation", wait_for_simulation)
builder.add_node("process_telemetry", process_telemetry)
builder.add_node("publish_state", publish_state)

# Define the flow
builder.add_edge("coordinate_sensors", "wait_for_simulation")
builder.add_edge("wait_for_simulation", "process_telemetry")
builder.add_edge("process_telemetry", "publish_state")
builder.add_edge("publish_state", END)
builder.set_entry_point("coordinate_sensors")

graph = builder.compile()

