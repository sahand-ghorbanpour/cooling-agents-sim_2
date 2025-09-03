import time, asyncio, os, requests, logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from common.metrics import (init_metrics, node_runs_total, node_duration_seconds, 
                          control_action, llm_request_duration_seconds)
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import get_latest_state, get_config, set_config
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("control-graph")
log = logging.getLogger("control-graph")

TARGET_TEMP = float(os.getenv("TARGET_TEMP_C", "24.0"))

async def get_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get current system state and control context"""
    gname="control_graph"; nname="get_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Get latest telemetry
        if "temps" not in state:
            latest = get_latest_state({})
            state.update(latest if latest else {})
        
        # Get control configuration
        state["control_config"] = {
            "target_temp_c": get_config("target_temp_c", TARGET_TEMP),
            "control_mode": get_config("control_mode", "hybrid_pid_llm"),
            "llm_coordination_enabled": get_config("llm_coordination_enabled", True),
            "emergency_override": get_config("emergency_override", False)
        }
        
        # Analyze current situation
        temps = state.get("temps", {})
        if temps:
            cabinet_temps = [v for k, v in temps.items() if k.startswith("cabinet_")]
            if cabinet_temps:
                avg_temp = sum(cabinet_temps) / len(cabinet_temps)
                temp_deviation = avg_temp - state["control_config"]["target_temp_c"]
                state["situation_analysis"] = {
                    "avg_temperature": avg_temp,
                    "temperature_deviation": temp_deviation,
                    "severity": _classify_severity(temp_deviation),
                    "requires_llm": abs(temp_deviation) > 1.0 or state["control_config"]["llm_coordination_enabled"]
                }
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def guardrails_validate(state: Dict[str, Any]) -> Dict[str, Any]:
    """Validate system state and control safety"""
    gname="control_graph"; nname="guardrails_validate"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        temps = state.get("temps", {})
        situation = state.get("situation_analysis", {})
        
        # Check guardrails
        try:
            guardrails_req = {
                "content": {
                    "temps": temps,
                    "situation": situation,
                    "target_temp": state.get("control_config", {}).get("target_temp_c", TARGET_TEMP)
                }
            }
            
            r = requests.post("http://guardrails:8001/validate", 
                            json=guardrails_req, timeout=2.0)
            
            if r.ok:
                result = r.json()
                state["guardrails_ok"] = result.get("ok", True)
                state["guardrails_issues"] = result.get("issues", [])
            else:
                log.warning(f"Guardrails service error: {r.status_code}")
                state["guardrails_ok"] = True  # Fail open
                state["guardrails_issues"] = []
                
        except Exception as e:
            log.warning(f"Guardrails validation failed: {e}")
            state["guardrails_ok"] = True  # Fail open in production
            state["guardrails_issues"] = []
        
        # Emergency override check
        if situation.get("severity") == "critical":
            log.warning("Critical temperature deviation detected - emergency protocols may activate")
            state["emergency_mode"] = True
        else:
            state["emergency_mode"] = False
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def llm_coordination(state: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinate with LLM for high-level control strategy"""
    gname="control_graph"; nname="llm_coordination"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        situation = state.get("situation_analysis", {})
        
        # Skip LLM if not needed or guardrails failed
        if not state.get("guardrails_ok", True) or not situation.get("requires_llm", False):
            state["llm_strategy"] = {"mode": "pid_only", "coordination": False}
            node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
            return state
        
        temps = state.get("temps", {})
        target_temp = state.get("control_config", {}).get("target_temp_c", TARGET_TEMP)
        
        try:
            # Request thermal optimization from MCP Math service
            t1 = time.time()
            mcp_request = {
                "temps": temps,
                "target_temp_c": target_temp,
                "energy_weight": get_config("energy_optimization_weight", 0.3),
                "stability_weight": get_config("stability_weight", 0.7),
                "current_actions": state.get("last_actions"),
                "constraints": {
                    "emergency_mode": state.get("emergency_mode", False),
                    "max_fan_speed": get_config("max_fan_speed_pct", 95.0)
                }
            }
            
            mcp_response = requests.post("http://mcp-math:7000/thermal_optimize", 
                                       json=mcp_request, timeout=10.0)
            
            if mcp_response.ok:
                optimization_result = mcp_response.json()
                
                # Also get LLM strategic input
                llm_request = {
                    "temps": temps,
                    "target": target_temp,
                    "optimization": optimization_result,
                    "situation": situation
                }
                
                llm_response = requests.post("http://llm-gateway:9000/plan", 
                                           json=llm_request, timeout=8.0)
                
                llm_duration = time.time() - t1
                model = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
                llm_request_duration_seconds.labels(model).observe(llm_duration)
                
                if llm_response.ok:
                    llm_result = llm_response.json()
                    
                    # Combine MCP optimization with LLM strategy
                    state["llm_strategy"] = {
                        "mode": "coordinated_control",
                        "mcp_optimization": optimization_result,
                        "llm_actions": llm_result.get("actions", {}),
                        "strategy": optimization_result.get("strategy", "proportional_control"),
                        "confidence": optimization_result.get("confidence", 0.8),
                        "reasoning": optimization_result.get("reasoning", "Thermal optimization applied")
                    }
                    
                    log.info(f"LLM coordination: strategy={optimization_result.get('strategy')}, "
                           f"confidence={optimization_result.get('confidence', 0):.2f}")
                else:
                    # Use MCP optimization only
                    state["llm_strategy"] = {
                        "mode": "mcp_only",
                        "mcp_optimization": optimization_result,
                        "strategy": optimization_result.get("strategy", "proportional_control"),
                        "confidence": optimization_result.get("confidence", 0.7)
                    }
            else:
                raise Exception(f"MCP Math service error: {mcp_response.status_code}")
                
        except Exception as e:
            log.warning(f"LLM/MCP coordination failed: {e}")
            # Fallback to simple coordination
            state["llm_strategy"] = {
                "mode": "fallback_coordination",
                "error": str(e),
                "fallback_actions": _generate_fallback_coordination(temps, target_temp)
            }
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def coordinate_runtime(state: Dict[str, Any]) -> Dict[str, Any]:
    """Send coordination signals to control runtime service"""
    gname="control_graph"; nname="coordinate_runtime"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        strategy = state.get("llm_strategy", {})
        control_config = state.get("control_config", {})
        
        # Prepare coordination message for runtime service
        coordination = {
            "type": "coordination",
            "timestamp": time.time(),
            "mode": strategy.get("mode", "pid_only"),
            "target_temp_c": control_config.get("target_temp_c", TARGET_TEMP),
            "emergency_mode": state.get("emergency_mode", False)
        }
        
        # Add LLM targets if available
        if "mcp_optimization" in strategy:
            mcp_result = strategy["mcp_optimization"]
            
            # Convert cabinet actions to target format
            llm_targets = {}
            cabinet_actions = mcp_result.get("cabinet_actions", {})
            
            for cabinet_key, action_list in cabinet_actions.items():
                # Convert cabinet action to fan percentage suggestion
                if len(action_list) >= 1:
                    sec_temp_adj = action_list[0]  # Secondary temperature adjustment
                    # Convert to fan percentage (rough mapping)
                    base_pct = 40.0
                    fan_pct = base_pct - sec_temp_adj * 40.0  # Negative adjustment = more cooling
                    fan_pct = max(10.0, min(95.0, fan_pct))
                    
                    legacy_key = cabinet_key.replace("cdu-cabinet-", "cabinet_") + "_fan_pct"
                    llm_targets[legacy_key] = fan_pct
            
            # Add cooling tower target
            ct_action = mcp_result.get("cooling_tower_action", 4)
            ct_pct = 10.0 + (ct_action / 8.0) * 80.0  # Map 0-8 to 10-90%
            llm_targets["cooling_tower_speed_pct"] = ct_pct
            
            coordination["llm_targets"] = llm_targets
            coordination["strategy"] = mcp_result.get("strategy", "unknown")
            coordination["confidence"] = mcp_result.get("confidence", 0.5)
        
        # Also include LLM direct actions if available
        if "llm_actions" in strategy:
            coordination["llm_direct_actions"] = strategy["llm_actions"]
        
        # Publish coordination to runtime service
        await nats_publish("dc.control.coordinate", coordination, agent="control_coordinator")
        
        # Also send targets to orchestrator topic for visibility
        if "llm_targets" in coordination:
            target_payload = {
                "targets": coordination["llm_targets"],
                "strategy": coordination.get("strategy", "coordinated"),
                "source": "control_graph"
            }
            await nats_publish(ROUTING["orch_targets"], target_payload, agent="control")
        
        state["coordination_sent"] = coordination
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def monitor_and_adapt(state: Dict[str, Any]) -> Dict[str, Any]:
    """Monitor control performance and adapt parameters"""
    gname="control_graph"; nname="monitor_and_adapt"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        situation = state.get("situation_analysis", {})
        strategy = state.get("llm_strategy", {})
        
        # Performance monitoring
        temp_deviation = situation.get("temperature_deviation", 0.0)
        severity = situation.get("severity", "normal")
        
        # Adaptive control parameters
        adaptations = {}
        
        if severity == "critical":
            # Emergency adaptations
            adaptations["control_loop_interval_ms"] = 250  # Faster control loop
            adaptations["llm_coordination_enabled"] = True  # Force LLM coordination
            adaptations["max_fan_speed_pct"] = 100.0  # Remove fan speed limits
            log.warning("Critical conditions - emergency adaptations activated")
            
        elif severity == "high":
            adaptations["control_loop_interval_ms"] = 400  # Slightly faster
            adaptations["llm_coordination_enabled"] = True
            
        elif severity == "low":
            # Energy optimization mode
            adaptations["control_loop_interval_ms"] = 800  # Slower, more stable
            adaptations["energy_optimization_weight"] = 0.4  # More energy focus
            
        else:  # normal
            # Reset to defaults
            adaptations["control_loop_interval_ms"] = 500
            adaptations["energy_optimization_weight"] = 0.3
        
        # Apply adaptations
        for key, value in adaptations.items():
            set_config(key, value)
            log.debug(f"Adapted {key} = {value}")
        
        # Update metrics for monitoring
        for i, temp in enumerate([v for k, v in state.get("temps", {}).items() 
                                 if k.startswith("cabinet_")], 1):
            control_action.labels(target=f"cabinet_{i}_temp").set(temp)
        
        state["adaptations"] = adaptations
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def _classify_severity(temp_deviation: float) -> str:
    """Classify temperature deviation severity"""
    abs_dev = abs(temp_deviation)
    if abs_dev > 4.0:
        return "critical"
    elif abs_dev > 2.0:
        return "high"
    elif abs_dev > 0.8:
        return "moderate"
    elif abs_dev > 0.3:
        return "low"
    else:
        return "normal"

def _generate_fallback_coordination(temps: Dict[str, float], target_temp: float) -> Dict[str, float]:
    """Generate fallback coordination when LLM/MCP services fail"""
    actions = {}
    
    for key, temp in temps.items():
        if key.startswith("cabinet_"):
            error = temp - target_temp
            # Simple proportional control
            fan_pct = max(10.0, min(90.0, 40.0 + error * 15.0))
            actions[key + "_fan_pct"] = fan_pct
    
    # Cooling tower
    if temps:
        avg_temp = sum(v for k, v in temps.items() if k.startswith("cabinet_")) / max(1, len([k for k in temps if k.startswith("cabinet_")]))
        error = avg_temp - target_temp
        ct_pct = max(15.0, min(85.0, 50.0 + error * 12.0))
        actions["cooling_tower_speed_pct"] = ct_pct
    
    return actions

# Build the control coordination graph
builder = StateGraph(dict)
builder.add_node("get_state", get_state)
builder.add_node("guardrails_validate", guardrails_validate)
builder.add_node("llm_coordination", llm_coordination)
builder.add_node("coordinate_runtime", coordinate_runtime)
builder.add_node("monitor_and_adapt", monitor_and_adapt)

# Define the coordination flow
builder.add_edge("get_state", "guardrails_validate")
builder.add_edge("guardrails_validate", "llm_coordination")
builder.add_edge("llm_coordination", "coordinate_runtime")
builder.add_edge("coordinate_runtime", "monitor_and_adapt")
builder.add_edge("monitor_and_adapt", END)
builder.set_entry_point("get_state")

graph = builder.compile()
