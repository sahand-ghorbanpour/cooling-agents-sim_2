import asyncio, json, os, logging, time, requests
from typing import Dict, Any, Optional
import numpy as np
from nats.aio.client import Client as NATS
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from common.logging import setup_json_logging
from common.config import get_config, set_config

log = setup_json_logging()
start_http_server(int(os.getenv("METRICS_PORT","9010")))

# Prometheus metrics
control_loops_total = Counter('control_loops_total', 'Total control loop executions')
control_loop_duration = Histogram('control_loop_duration_seconds', 'Control loop execution time')
control_action_applied = Gauge('control_action_applied', 'Applied control action', ['target', 'action_type'])
control_error = Gauge('control_error_celsius', 'Control error from target temperature')
llm_override_count = Counter('llm_override_total', 'Times LLM overrode PID control')

# Control state
class ControlState:
    def __init__(self):
        self.target_temp_c = float(os.getenv("TARGET_TEMP_C", "24.0"))
        self.current_temps = {}
        self.last_actions = {}
        self.control_history = []
        self.llm_targets = {}
        self.pid_errors = {}
        self.integral_terms = {}
        self.last_time = time.time()
        
        # PID parameters
        self.kp = float(os.getenv("PID_KP", "0.8"))
        self.ki = float(os.getenv("PID_KI", "0.1"))
        self.kd = float(os.getenv("PID_KD", "0.05"))
        self.max_integral = float(os.getenv("PID_MAX_INTEGRAL", "10.0"))
        
    def reset_integral(self, cabinet_id: str):
        """Reset integral term for anti-windup"""
        self.integral_terms[cabinet_id] = 0.0

state = ControlState()

async def main():
    """Main control runtime loop"""
    nc = NATS()
    await nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
    
    # Subscribe to telemetry updates
    await nc.subscribe("dc.telemetry.state", cb=handle_telemetry)
    await nc.subscribe("simulation.state", cb=handle_telemetry)  # Compatibility
    
    # Subscribe to LLM coordination targets
    await nc.subscribe("dc.orch.targets", cb=handle_llm_targets)
    await nc.subscribe("dc.control.targets", cb=handle_llm_targets)
    
    # Subscribe to control graph coordination
    await nc.subscribe("dc.control.coordinate", cb=handle_coordination)
    
    log.info("Control runtime started - subscriptions active")
    
    # Main control loop
    while True:
        try:
            await run_control_loop(nc)
            
            # Control frequency based on config
            loop_interval = float(get_config("control_loop_interval_ms", 500)) / 1000.0
            await asyncio.sleep(loop_interval)
            
        except Exception as e:
            log.error(f"Control loop error: {e}")
            await asyncio.sleep(1.0)

async def handle_telemetry(msg):
    """Handle incoming sensor telemetry"""
    try:
        payload = json.loads(msg.data.decode())
        telemetry = payload.get("data", {})
        temps = telemetry.get("temps", {})
        
        if temps:
            state.current_temps.update(temps)
            log.debug(f"Updated temperatures: {len(temps)} sensors")
            
    except Exception as e:
        log.error(f"Error handling telemetry: {e}")

async def handle_llm_targets(msg):
    """Handle LLM-generated target suggestions"""
    try:
        payload = json.loads(msg.data.decode())
        targets = payload.get("targets", {})
        
        if targets:
            state.llm_targets.update(targets)
            log.info(f"Received LLM targets: {list(targets.keys())}")
            
    except Exception as e:
        log.error(f"Error handling LLM targets: {e}")

async def handle_coordination(msg):
    """Handle coordination messages from control graph"""
    try:
        payload = json.loads(msg.data.decode())
        command = payload.get("command")
        
        if command == "update_target":
            new_target = float(payload.get("target_temp_c", state.target_temp_c))
            state.target_temp_c = new_target
            log.info(f"Updated target temperature to {new_target}°C")
            
        elif command == "reset_integral":
            cabinet_id = payload.get("cabinet_id")
            if cabinet_id:
                state.reset_integral(cabinet_id)
            else:
                # Reset all
                state.integral_terms.clear()
            log.info(f"Reset integral terms for {cabinet_id or 'all cabinets'}")
            
        elif command == "update_pid":
            # Update PID parameters
            state.kp = float(payload.get("kp", state.kp))
            state.ki = float(payload.get("ki", state.ki))
            state.kd = float(payload.get("kd", state.kd))
            log.info(f"Updated PID parameters: Kp={state.kp}, Ki={state.ki}, Kd={state.kd}")
            
    except Exception as e:
        log.error(f"Error handling coordination: {e}")

async def run_control_loop(nc: NATS):
    """Execute the main control loop"""
    with control_loop_duration.time():
        control_loops_total.inc()
        current_time = time.time()
        dt = current_time - state.last_time
        state.last_time = current_time
        
        if not state.current_temps:
            log.warning("No temperature data available for control")
            return
        
        # Generate control actions
        actions = await generate_control_actions(dt)
        
        if actions:
            # Convert to simulation format and publish
            sim_actions = convert_to_simulation_actions(actions)
            
            # Publish to simulation
            sim_payload = {
                "type": "control.actions",
                "data": sim_actions,
                "timestamp": current_time,
                "source": "control_runtime"
            }
            
            await nc.publish("simulation.actions", json.dumps(sim_payload).encode())
            
            # Also notify sensor graph directly (for hybrid architecture)
            try:
                from graphs.sensor_graph import inject_control_actions
                inject_control_actions(sim_actions)
            except ImportError:
                log.warning("Could not inject actions directly to sensor graph")
            
            # Publish control status for monitoring
            status_payload = {
                "type": "control.status",
                "data": {
                    "actions": actions,
                    "errors": state.pid_errors,
                    "target_temp": state.target_temp_c,
                    "control_mode": "pid_with_llm_coordination"
                },
                "timestamp": current_time
            }
            
            await nc.publish("dc.control.status", json.dumps(status_payload).encode())
            
            # Store for history
            state.last_actions = actions.copy()
            state.control_history.append({
                "timestamp": current_time,
                "actions": actions.copy(),
                "errors": state.pid_errors.copy()
            })
            
            # Trim history
            if len(state.control_history) > 100:
                state.control_history = state.control_history[-100:]

async def generate_control_actions(dt: float) -> Dict[str, Any]:
    """Generate control actions using PID + LLM coordination"""
    actions = {}
    
    # Calculate global temperature and error
    cabinet_temps = {k: v for k, v in state.current_temps.items() if k.startswith("cabinet_")}
    if not cabinet_temps:
        return actions
    
    global_temp = sum(cabinet_temps.values()) / len(cabinet_temps)
    global_error = global_temp - state.target_temp_c
    control_error.set(global_error)
    
    # Generate cabinet control actions
    for cabinet_id, temp in cabinet_temps.items():
        action = await generate_cabinet_action(cabinet_id, temp, dt)
        if action is not None:
            actions[cabinet_id] = action
    
    # Generate cooling tower action
    ct_action = await generate_cooling_tower_action(global_temp, global_error)
    if ct_action is not None:
        actions["cooling_tower"] = ct_action
    
    return actions

async def generate_cabinet_action(cabinet_id: str, current_temp: float, dt: float) -> Optional[Dict[str, float]]:
    """Generate PID-based cabinet control action with LLM coordination"""
    
    error = current_temp - state.target_temp_c
    state.pid_errors[cabinet_id] = error
    
    # Initialize integral term if needed
    if cabinet_id not in state.integral_terms:
        state.integral_terms[cabinet_id] = 0.0
    
    # PID calculation
    proportional = error
    state.integral_terms[cabinet_id] += error * dt
    
    # Anti-windup
    state.integral_terms[cabinet_id] = np.clip(
        state.integral_terms[cabinet_id], 
        -state.max_integral, 
        state.max_integral
    )
    
    integral = state.integral_terms[cabinet_id]
    
    # Derivative term
    prev_error = 0.0
    if state.control_history:
        prev_errors = state.control_history[-1].get("errors", {})
        prev_error = prev_errors.get(cabinet_id, 0.0)
    
    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    
    # PID output
    pid_output = state.kp * proportional + state.ki * integral + state.kd * derivative
    
    # Convert PID output to fan percentage (0-100%)
    base_fan_pct = 30.0  # Base fan speed
    fan_pct = base_fan_pct + pid_output * 20.0  # Scale PID output
    fan_pct = np.clip(fan_pct, 5.0, 100.0)
    
    # Check for LLM override
    llm_key = f"{cabinet_id}_fan_pct"
    if llm_key in state.llm_targets:
        llm_pct = float(state.llm_targets[llm_key])
        
        # Use LLM suggestion if it's reasonable (not too different from PID)
        if abs(llm_pct - fan_pct) < 30.0:  # Within 30% of PID suggestion
            fan_pct = llm_pct
            llm_override_count.inc()
            log.debug(f"Using LLM target for {cabinet_id}: {llm_pct}%")
    
    # Update metrics
    control_action_applied.labels(target=cabinet_id, action_type="fan_pct").set(fan_pct)
    
    return {
        "fan_pct": fan_pct,
        "pid_output": pid_output,
        "error": error,
        "p_term": proportional,
        "i_term": integral,
        "d_term": derivative
    }

async def generate_cooling_tower_action(global_temp: float, global_error: float) -> Optional[Dict[str, Any]]:
    """Generate cooling tower control action"""
    
    # Base cooling tower speed using simple proportional control
    base_speed = 40.0  # Base speed percentage
    speed_adjustment = global_error * 15.0  # Scale factor
    ct_speed_pct = base_speed + speed_adjustment
    ct_speed_pct = np.clip(ct_speed_pct, 10.0, 100.0)
    
    # Check for LLM coordination
    if "cooling_tower_speed_pct" in state.llm_targets:
        llm_speed = float(state.llm_targets["cooling_tower_speed_pct"])
        if abs(llm_speed - ct_speed_pct) < 40.0:  # Within reasonable range
            ct_speed_pct = llm_speed
            llm_override_count.inc()
    
    # Convert to discrete action for simulation compatibility (0-8)
    discrete_action = int(np.clip((ct_speed_pct - 10.0) / 11.25, 0, 8))  # Map 10-100% to 0-8
    
    # Update metrics
    control_action_applied.labels(target="cooling_tower", action_type="speed_pct").set(ct_speed_pct)
    control_action_applied.labels(target="cooling_tower", action_type="discrete").set(discrete_action)
    
    return {
        "speed_pct": ct_speed_pct,
        "discrete_action": discrete_action,
        "error": global_error
    }

def convert_to_simulation_actions(actions: Dict[str, Any]) -> Dict[str, Any]:
    """Convert control actions to simulation format"""
    sim_actions = {}
    
    # Convert cabinet actions
    for i in range(1, 6):
        cabinet_key = f"cabinet_{i}"
        sim_key = f"cdu-cabinet-{i}"
        
        if cabinet_key in actions:
            action_data = actions[cabinet_key]
            fan_pct = action_data["fan_pct"]
            
            # Convert fan percentage to simulation action format
            # [sec_temp, pressure, valve1, valve2, valve3]
            sec_temp = (fan_pct - 50.0) / 100.0  # Normalize to [-0.5, 0.5]
            pressure = 0.1  # Default pressure
            
            # Valve positions based on fan speed (higher speed = more flow)
            primary_flow = min(0.5, fan_pct / 200.0 + 0.2)
            secondary_flow = min(0.4, fan_pct / 250.0 + 0.1)
            bypass_flow = 1.0 - primary_flow - secondary_flow
            
            sim_actions[sim_key] = np.array([
                sec_temp,
                pressure, 
                primary_flow,
                secondary_flow,
                bypass_flow
            ], dtype=np.float32)
    
    # Convert cooling tower action
    if "cooling_tower" in actions:
        ct_data = actions["cooling_tower"]
        sim_actions["cooling-tower-1"] = ct_data["discrete_action"]
    
    return sim_actions

if __name__=="__main__":
    asyncio.run(main())