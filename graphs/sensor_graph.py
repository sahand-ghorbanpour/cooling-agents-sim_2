import time, asyncio, logging, os
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

# Import your real physics simulation
from env.frontier_env import SmallFrontierModel
import numpy as np

# Global simulation environment - initialized once, reused
_env = None
_current_actions = None
_step_count = 0

def initialize_simulation():
    """Initialize the FMU-based simulation environment"""
    global _env
    if _env is None:
        try:
            _env = SmallFrontierModel(
                start_time=0,
                stop_time=86400,  # 24 hours
                step_size=15.0,   # 15 seconds
                use_reward_shaping='reward_shaping_v1'
            )
            # Reset to get initial state
            initial_state, _ = _env.reset()
            log.info("SmallFrontierModel initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize SmallFrontierModel: {e}")
            _env = None
    return _env

async def step_env(state: Dict[str, Any]) -> Dict[str, Any]:
    """Step the real physics simulation and extract telemetry"""
    gname="sensor_graph"; nname="step_env"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        global _env, _current_actions, _step_count
        
        # Initialize simulation if needed
        if _env is None:
            _env = initialize_simulation()
        
        temps = {}
        meta = {}
        rewards = {}
        
        try:
            if _env is not None:
                # Use actions from control if available, otherwise default actions
                if _current_actions is not None:
                    actions = _current_actions
                    _current_actions = None  # Reset after use
                else:
                    # Default actions - maintain current state
                    actions = {
                        'cdu-cabinet-1': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-2': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-3': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-4': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-5': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cooling-tower-1': 4  # Maintain
                    }
                
                # Step simulation
                obs, rewards_dict, terminateds, truncateds, info = _env.step(actions)
                
                # Extract temperature data
                for i in range(1, 6):
                    cabinet_key = f"cdu-cabinet-{i}"
                    if cabinet_key in obs:
                        # Convert normalized observations to Celsius
                        cabinet_obs = obs[cabinet_key]
                        if len(cabinet_obs) >= 3:
                            # First 3 elements are boundary temperatures (normalized)
                            temp_norm = cabinet_obs[:3]
                            # Convert from normalized [-1,1] to Kelvin, then Celsius
                            temp_k = ((temp_norm + 1) / 2) * (313.15 - 293.15) + 293.15
                            avg_temp_c = float(np.mean(temp_k) - 273.15)
                            temps[f"cabinet_{i}"] = avg_temp_c
                
                # Extract cooling tower data
                if "cooling-tower-1" in obs:
                    ct_obs = obs["cooling-tower-1"]
                    if len(ct_obs) >= 3:
                        # Water temperature
                        water_temp_norm = ct_obs[2]
                        water_temp_k = ((water_temp_norm + 1) / 2) * (313.15 - 293.15) + 293.15
                        temps["cooling_tower"] = float(water_temp_k - 273.15)
                
                # Calculate energy consumption (simplified)
                total_energy = 0.0
                if "cooling-tower-1" in obs:
                    ct_obs = obs["cooling-tower-1"]
                    if len(ct_obs) >= 2:
                        fan_power = np.sum(ct_obs[:2])  # Fan powers
                        # Convert normalized fan power to kW (rough estimate)
                        total_energy = float((fan_power + 2) / 4 * 150)  # 150kW max estimate
                
                meta.update({
                    "energy_kw": total_energy,
                    "step_count": _step_count,
                    "rewards": {k: float(v) for k, v in rewards_dict.items()},
                    "simulation_time": _env.current_time if hasattr(_env, 'current_time') else _step_count * 15.0
                })
                
                _step_count += 1
                
            else:
                # Fallback mock data if simulation failed
                temps = {f"cabinet_{i}": 25.0 + np.random.normal(0, 0.5) for i in range(1,6)}
                temps["cooling_tower"] = 22.0
                meta = {"energy_kw": 100.0, "step_count": _step_count}
                
        except Exception as e:
            log.error(f"Simulation step failed: {e}")
            # Emergency fallback
            temps = {f"cabinet_{i}": 25.0 for i in range(1,6)}
            temps["cooling_tower"] = 22.0
            meta = {"energy_kw": 100.0, "step_count": _step_count, "error": str(e)}
        
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
            
            # Efficiency score: better when temperature closer to target (24°C) and energy lower
            temp_efficiency = max(0.0, min(1.0, 1.0 - abs(avg_temp - 24.0) / 10.0))
            energy_efficiency = max(0.0, min(1.0, 1.0 - energy / 200.0))
            overall_efficiency = (temp_efficiency + energy_efficiency) / 2.0
            efficiency_score.set(overall_efficiency)
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return {"temps": temps, "meta": meta, "ts": time.time()}

async def receive_actions(state: Dict[str, Any]) -> Dict[str, Any]:
    """Receive and prepare actions for next simulation step"""
    gname="sensor_graph"; nname="receive_actions"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        global _current_actions
        
        # Check for incoming actions from control system
        # This would be set by the NATS listener or control coordination
        actions_received = state.get("control_actions")
        if actions_received:
            _current_actions = actions_received
            log.info(f"Received control actions for next simulation step")
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def publish_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Publish sensor telemetry to NATS"""
    gname="sensor_graph"; nname="publish_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Prepare telemetry payload
        payload = {
            "type": "telemetry.state",
            "data": state,
            "timestamp": time.time(),
            "source": "sensor_graph"
        }
        
        # Publish to multiple topics for different consumers
        await nats_publish(ROUTING["state_out"], payload, agent="sensor")
        await nats_publish("simulation.state", payload, agent="sensor")  # For compatibility
        
        # Store latest state for other services
        set_latest_state(state)
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

# Create the sensor graph
builder = StateGraph(dict)
builder.add_node("receive_actions", receive_actions)
builder.add_node("step_env", step_env)
builder.add_node("publish_state", publish_state)

# Define the flow
builder.add_edge("receive_actions", "step_env")
builder.add_edge("step_env", "publish_state")
builder.add_edge("publish_state", END)
builder.set_entry_point("receive_actions")

graph = builder.compile()

# Function to inject actions from external control services
def inject_control_actions(actions):
    """Called by control runtime to inject actions for next simulation step"""
    global _current_actions
    _current_actions = actions
    log.info(f"Injected control actions: {list(actions.keys()) if actions else 'None'}")
    