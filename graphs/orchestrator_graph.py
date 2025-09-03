import time, asyncio, requests, logging, os
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from common.metrics import (init_metrics, node_runs_total, node_duration_seconds, 
                          nats_messages_total)
from common.nats_utils import publish as nats_publish, ROUTING
from common.config import get_latest_state, get_config, set_config
from common.otel import init_tracer
from common.logging import setup_json_logging

setup_json_logging()
init_metrics()
tracer = init_tracer("orchestrator-graph")
log = logging.getLogger("orchestrator-graph")

# Orchestration strategies
COORDINATION_STRATEGIES = {
    "autonomous": "Let individual agents operate independently with minimal coordination",
    "centralized": "Central orchestrator makes all high-level decisions",
    "hybrid": "Combine autonomous operation with strategic coordination",
    "emergency": "Override autonomous operation for emergency response"
}

def fetch_system_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch comprehensive system state from all sources"""
    gname="orchestrator_graph"; nname="fetch_system_state"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Get latest telemetry from sensor systems
        latest_state = get_latest_state({})
        
        # Get control system status
        control_status = get_config("last_control_status", {})
        
        # Get agent performance metrics
        agent_metrics = {
            "sensor_health": _check_agent_health("sensor"),
            "control_health": _check_agent_health("control"),
            "llm_health": _check_agent_health("llm-gateway"),
            "mcp_health": _check_agent_health("mcp-math")
        }
        
        # Analyze system-wide performance
        system_analysis = _analyze_system_performance(latest_state, control_status, agent_metrics)
        
        state.update({
            "latest_state": latest_state,
            "control_status": control_status,
            "agent_metrics": agent_metrics,
            "system_analysis": system_analysis,
            "coordination_mode": get_config("coordination_mode", "hybrid"),
            "emergency_mode": get_config("emergency_override", False)
        })
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def assess_coordination_need(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assess whether system-level coordination is needed"""
    gname="orchestrator_graph"; nname="assess_coordination_need"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        system_analysis = state.get("system_analysis", {})
        agent_metrics = state.get("agent_metrics", {})
        latest_state = state.get("latest_state", {})
        
        coordination_triggers = []
        coordination_priority = "low"
        
        # Check for performance issues
        if system_analysis.get("performance_score", 1.0) < 0.6:
            coordination_triggers.append("low_system_performance")
            coordination_priority = "high"
        
        # Check for agent failures
        failed_agents = [agent for agent, health in agent_metrics.items() 
                        if not health.get("healthy", True)]
        if failed_agents:
            coordination_triggers.append(f"agent_failures_{len(failed_agents)}")
            coordination_priority = "high" if len(failed_agents) > 1 else "medium"
        
        # Check for temperature anomalies
        temps = latest_state.get("temps", {})
        if temps:
            cabinet_temps = [v for k, v in temps.items() if k.startswith("cabinet_")]
            if cabinet_temps:
                temp_spread = max(cabinet_temps) - min(cabinet_temps)
                avg_temp = sum(cabinet_temps) / len(cabinet_temps)
                target_temp = float(get_config("target_temp_c", 24.0))
                
                if temp_spread > 5.0:  # Large temperature variation
                    coordination_triggers.append("high_temp_spread")
                    coordination_priority = max(coordination_priority, "medium")
                
                if abs(avg_temp - target_temp) > 3.0:  # Large deviation from target
                    coordination_triggers.append("high_temp_deviation")  
                    coordination_priority = "high"
        
        # Check for resource constraints
        if system_analysis.get("energy_efficiency", 1.0) < 0.4:
            coordination_triggers.append("low_energy_efficiency")
            coordination_priority = max(coordination_priority, "medium")
        
        # Check for communication issues
        if system_analysis.get("communication_health", 1.0) < 0.8:
            coordination_triggers.append("communication_degradation")
            coordination_priority = max(coordination_priority, "medium")
        
        # Determine coordination strategy
        coordination_needed = len(coordination_triggers) > 0
        coordination_strategy = _select_coordination_strategy(
            coordination_triggers, 
            coordination_priority,
            state.get("coordination_mode", "hybrid")
        )
        
        state.update({
            "coordination_needed": coordination_needed,
            "coordination_triggers": coordination_triggers,
            "coordination_priority": coordination_priority,
            "coordination_strategy": coordination_strategy
        })
        
        log.info(f"Coordination assessment: needed={coordination_needed}, "
                f"priority={coordination_priority}, triggers={coordination_triggers}")
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def strategic_planning(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate strategic coordination plan"""
    gname="orchestrator_graph"; nname="strategic_planning"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        if not state.get("coordination_needed", False):
            state["strategic_plan"] = {"action": "monitor", "interventions": []}
            node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
            return state
        
        coordination_strategy = state.get("coordination_strategy", "hybrid")
        coordination_triggers = state.get("coordination_triggers", [])
        system_analysis = state.get("system_analysis", {})
        
        # Generate strategic interventions based on triggers
        interventions = []
        
        if "low_system_performance" in coordination_triggers:
            interventions.extend([
                {"type": "tuning", "target": "control_parameters", "priority": "high"},
                {"type": "coordination", "target": "agent_synchronization", "priority": "high"}
            ])
        
        if "high_temp_deviation" in coordination_triggers:
            interventions.extend([
                {"type": "control", "target": "emergency_cooling", "priority": "critical"},
                {"type": "llm", "target": "thermal_strategy_update", "priority": "high"}
            ])
        
        if "high_temp_spread" in coordination_triggers:
            interventions.extend([
                {"type": "balancing", "target": "cabinet_load_distribution", "priority": "medium"},
                {"type": "optimization", "target": "valve_position_rebalance", "priority": "medium"}
            ])
        
        if any("agent_failure" in trigger for trigger in coordination_triggers):
            interventions.extend([
                {"type": "failover", "target": "backup_control_activation", "priority": "critical"},
                {"type": "monitoring", "target": "enhanced_health_checks", "priority": "high"}
            ])
        
        if "low_energy_efficiency" in coordination_triggers:
            interventions.extend([
                {"type": "optimization", "target": "energy_efficiency_mode", "priority": "low"},
                {"type": "scheduling", "target": "load_shifting", "priority": "low"}
            ])
        
        # Add LLM-based strategic analysis if available
        llm_strategy = None
        try:
            if system_analysis.get("complexity_score", 0.5) > 0.7:  # Complex situation
                llm_strategy = await _request_llm_strategic_analysis(state)
        except Exception as e:
            log.warning(f"LLM strategic analysis failed: {e}")
        
        strategic_plan = {
            "action": "coordinate" if interventions else "monitor",
            "strategy": coordination_strategy,
            "interventions": sorted(interventions, key=lambda x: _priority_score(x["priority"]), reverse=True),
            "llm_strategy": llm_strategy,
            "execution_timeline": _generate_execution_timeline(interventions),
            "success_criteria": _define_success_criteria(coordination_triggers),
            "rollback_plan": _generate_rollback_plan(interventions)
        }
        
        state["strategic_plan"] = strategic_plan
        
        log.info(f"Strategic plan generated: {len(interventions)} interventions, "
                f"strategy={coordination_strategy}")
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def execute_coordination(state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the strategic coordination plan"""
    gname="orchestrator_graph"; nname="execute_coordination" 
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        strategic_plan = state.get("strategic_plan", {})
        
        if strategic_plan.get("action") != "coordinate":
            state["execution_results"] = {"action": "no_coordination_needed"}
            node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
            return state
        
        interventions = strategic_plan.get("interventions", [])
        execution_results = []
        
        # Execute interventions in priority order
        for intervention in interventions:
            try:
                result = await _execute_intervention(intervention, state)
                execution_results.append(result)
                
                # Stop if critical intervention fails
                if intervention["priority"] == "critical" and not result.get("success", False):
                    log.error(f"Critical intervention failed: {intervention}")
                    break
                    
            except Exception as e:
                log.error(f"Intervention execution failed: {intervention}, error: {e}")
                execution_results.append({
                    "intervention": intervention,
                    "success": False,
                    "error": str(e)
                })
        
        # Update system configuration based on successful interventions
        successful_interventions = [r for r in execution_results if r.get("success", False)]
        if successful_interventions:
            await _update_system_configuration(successful_interventions)
        
        state["execution_results"] = {
            "total_interventions": len(interventions),
            "successful_interventions": len(successful_interventions),
            "failed_interventions": len(interventions) - len(successful_interventions),
            "details": execution_results
        }
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

def monitor_and_adapt(state: Dict[str, Any]) -> Dict[str, Any]:
    """Monitor coordination results and adapt system behavior"""
    gname="orchestrator_graph"; nname="monitor_and_adapt"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        strategic_plan = state.get("strategic_plan", {})
        execution_results = state.get("execution_results", {})
        
        # Evaluate coordination effectiveness
        effectiveness_score = _evaluate_coordination_effectiveness(
            strategic_plan, 
            execution_results,
            state.get("system_analysis", {})
        )
        
        # Update coordination history for learning
        coordination_history = get_config("coordination_history", [])
        coordination_record = {
            "timestamp": time.time(),
            "triggers": state.get("coordination_triggers", []),
            "strategy": strategic_plan.get("strategy"),
            "interventions": len(strategic_plan.get("interventions", [])),
            "effectiveness_score": effectiveness_score,
            "success_rate": execution_results.get("successful_interventions", 0) / max(1, execution_results.get("total_interventions", 1))
        }
        coordination_history.append(coordination_record)
        
        # Keep last 100 coordination records
        if len(coordination_history) > 100:
            coordination_history = coordination_history[-100:]
        set_config("coordination_history", coordination_history)
        
        # Adapt coordination parameters based on historical performance
        adaptations = _generate_adaptations(coordination_history, effectiveness_score)
        for key, value in adaptations.items():
            set_config(key, value)
            log.info(f"Adapted coordination parameter: {key} = {value}")
        
        # Publish coordination status
        coordination_status = {
            "timestamp": time.time(),
            "coordination_active": state.get("coordination_needed", False),
            "effectiveness_score": effectiveness_score,
            "system_health": state.get("system_analysis", {}).get("performance_score", 1.0),
            "active_interventions": len([r for r in execution_results.get("details", []) if r.get("success")]),
            "next_assessment": time.time() + get_config("coordination_interval_seconds", 300)
        }
        
        asyncio.run(nats_publish("dc.orchestrator.status", coordination_status, agent="orchestrator"))
        
        state.update({
            "coordination_effectiveness": effectiveness_score,
            "adaptations": adaptations,
            "coordination_status": coordination_status
        })
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

# Helper functions
def _check_agent_health(agent_name: str) -> Dict[str, Any]:
    """Check health of a specific agent"""
    # This would typically check metrics, recent activity, etc.
    return {
        "healthy": True,  # Simplified for now
        "last_activity": time.time(),
        "error_rate": 0.0,
        "response_time": 0.1
    }

def _analyze_system_performance(latest_state: Dict, control_status: Dict, agent_metrics: Dict) -> Dict[str, Any]:
    """Analyze overall system performance"""
    performance_score = 1.0
    energy_efficiency = 1.0
    communication_health = 1.0
    
    # Analyze temperature performance
    temps = latest_state.get("temps", {})
    if temps:
        cabinet_temps = [v for k, v in temps.items() if k.startswith("cabinet_")]
        if cabinet_temps:
            target_temp = 24.0
            avg_deviation = sum(abs(t - target_temp) for t in cabinet_temps) / len(cabinet_temps)
            performance_score *= max(0.0, 1.0 - avg_deviation / 10.0)
    
    # Analyze agent health
    healthy_agents = sum(1 for health in agent_metrics.values() if health.get("healthy", False))
    total_agents = len(agent_metrics)
    if total_agents > 0:
        communication_health = healthy_agents / total_agents
        performance_score *= communication_health
    
    return {
        "performance_score": performance_score,
        "energy_efficiency": energy_efficiency,  
        "communication_health": communication_health,
        "complexity_score": 0.5  # Would calculate based on system state complexity
    }

def _select_coordination_strategy(triggers: List[str], priority: str, mode: str) -> str:
    """Select appropriate coordination strategy"""
    if priority == "critical" or "emergency" in mode:
        return "emergency"
    elif priority == "high":
        return "centralized"
    elif mode == "autonomous":
        return "autonomous"
    else:
        return "hybrid"

async def _request_llm_strategic_analysis(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Request strategic analysis from LLM"""
    try:
        # Prepare context for LLM
        context = {
            "system_state": state.get("latest_state", {}),
            "coordination_triggers": state.get("coordination_triggers", []),
            "agent_health": state.get("agent_metrics", {}),
            "performance_analysis": state.get("system_analysis", {})
        }
        
        # Request strategic analysis
        response = requests.post(
            "http://llm-gateway:9000/strategic_analysis",
            json={"context": context, "request_type": "coordination_strategy"},
            timeout=10.0
        )
        
        if response.ok:
            return response.json()
        else:
            return None
            
    except Exception as e:
        log.warning(f"LLM strategic analysis request failed: {e}")
        return None

def _priority_score(priority: str) -> int:
    """Convert priority string to numeric score"""
    return {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(priority, 0)

def _generate_execution_timeline(interventions: List[Dict]) -> Dict[str, Any]:
    """Generate execution timeline for interventions"""
    timeline = {}
    current_time = time.time()
    
    for i, intervention in enumerate(interventions):
        priority = intervention["priority"]
        delay = {"critical": 0, "high": 5, "medium": 30, "low": 60}.get(priority, 60)
        timeline[f"intervention_{i}"] = current_time + delay
    
    return timeline

def _define_success_criteria(triggers: List[str]) -> Dict[str, Any]:
    """Define success criteria for coordination"""
    criteria = {}
    
    if "high_temp_deviation" in triggers:
        criteria["temperature_stable"] = "avg_deviation < 1.0"
    if "low_system_performance" in triggers:
        criteria["performance_improved"] = "performance_score > 0.8"
    if "low_energy_efficiency" in triggers:
        criteria["efficiency_improved"] = "energy_efficiency > 0.7"
    
    return criteria

def _generate_rollback_plan(interventions: List[Dict]) -> Dict[str, Any]:
    """Generate rollback plan in case coordination fails"""
    return {
        "rollback_timeout": 300,  # 5 minutes
        "rollback_triggers": ["performance_degradation", "system_instability"],
        "rollback_actions": ["restore_previous_config", "activate_failsafe_mode"]
    }

async def _execute_intervention(intervention: Dict, state: Dict) -> Dict[str, Any]:
    """Execute a specific intervention"""
    intervention_type = intervention["type"]
    target = intervention["target"]
    
    try:
        if intervention_type == "control":
            return await _execute_control_intervention(target, state)
        elif intervention_type == "optimization":
            return await _execute_optimization_intervention(target, state)
        elif intervention_type == "coordination":
            return await _execute_coordination_intervention(target, state)
        elif intervention_type == "tuning":
            return await _execute_tuning_intervention(target, state)
        else:
            return {"success": False, "error": f"Unknown intervention type: {intervention_type}"}
            
    except Exception as e:
        return {"success": False, "error": str(e), "intervention": intervention}

async def _execute_control_intervention(target: str, state: Dict) -> Dict[str, Any]:
    """Execute control-related interventions"""
    if target == "emergency_cooling":
        # Send emergency cooling command
        emergency_command = {
            "command": "emergency_cooling",
            "priority": "critical",
            "target_temp_override": 20.0,
            "max_fan_speed": 100.0
        }
        await nats_publish("dc.control.emergency", emergency_command, agent="orchestrator")
        return {"success": True, "action": "emergency_cooling_activated"}
    
    return {"success": False, "error": f"Unknown control target: {target}"}

async def _execute_optimization_intervention(target: str, state: Dict) -> Dict[str, Any]:
    """Execute optimization interventions"""
    if target == "energy_efficiency_mode":
        # Enable energy efficiency mode
        set_config("energy_optimization_weight", 0.6)
        set_config("control_aggressiveness", 0.7)
        return {"success": True, "action": "energy_efficiency_enabled"}
    
    return {"success": False, "error": f"Unknown optimization target: {target}"}

async def _execute_coordination_intervention(target: str, state: Dict) -> Dict[str, Any]:
    """Execute coordination interventions"""
    if target == "agent_synchronization":
        # Trigger agent synchronization
        sync_command = {"command": "synchronize", "timestamp": time.time()}
        await nats_publish("dc.agents.sync", sync_command, agent="orchestrator")
        return {"success": True, "action": "agent_synchronization_triggered"}
    
    return {"success": False, "error": f"Unknown coordination target: {target}"}

async def _execute_tuning_intervention(target: str, state: Dict) -> Dict[str, Any]:
    """Execute parameter tuning interventions"""
    if target == "control_parameters":
        # Adjust control parameters based on current performance
        performance_score = state.get("system_analysis", {}).get("performance_score", 1.0)
        if performance_score < 0.7:
            # Increase control aggressiveness
            new_kp = get_config("PID_KP", 0.8) * 1.2
            set_config("PID_KP", min(2.0, new_kp))
            return {"success": True, "action": f"increased_control_gain_to_{new_kp:.2f}"}
    
    return {"success": False, "error": f"Unknown tuning target: {target}"}

async def _update_system_configuration(successful_interventions: List[Dict]) -> None:
    """Update system configuration based on successful interventions"""
    config_updates = {}
    
    for result in successful_interventions:
        action = result.get("action", "")
        if "emergency_cooling" in action:
            config_updates["emergency_mode"] = True
        elif "energy_efficiency" in action:
            config_updates["energy_mode"] = True
        elif "control_gain" in action:
            config_updates["control_tuned"] = True
    
    # Apply configuration updates
    for key, value in config_updates.items():
        set_config(key, value)

def _evaluate_coordination_effectiveness(strategic_plan: Dict, execution_results: Dict, system_analysis: Dict) -> float:
    """Evaluate how effective the coordination was"""
    base_score = 0.5
    
    # Success rate component
    success_rate = execution_results.get("successful_interventions", 0) / max(1, execution_results.get("total_interventions", 1))
    base_score += success_rate * 0.3
    
    # System performance component  
    performance_score = system_analysis.get("performance_score", 0.5)
    base_score += performance_score * 0.2
    
    return min(1.0, base_score)

def _generate_adaptations(coordination_history: List[Dict], current_effectiveness: float) -> Dict[str, Any]:
    """Generate adaptive changes based on coordination history"""
    adaptations = {}
    
    if len(coordination_history) >= 5:
        recent_effectiveness = [record["effectiveness_score"] for record in coordination_history[-5:]]
        avg_effectiveness = sum(recent_effectiveness) / len(recent_effectiveness)
        
        if avg_effectiveness < 0.6:
            # Increase coordination frequency if effectiveness is low
            current_interval = get_config("coordination_interval_seconds", 300)
            adaptations["coordination_interval_seconds"] = max(60, int(current_interval * 0.8))
        elif avg_effectiveness > 0.9:
            # Decrease frequency if very effective
            current_interval = get_config("coordination_interval_seconds", 300)
            adaptations["coordination_interval_seconds"] = min(600, int(current_interval * 1.2))
    
    return adaptations

# Build the orchestrator graph
builder = StateGraph(dict)
builder.add_node("fetch_system_state", fetch_system_state)
builder.add_node("assess_coordination_need", assess_coordination_need)
builder.add_node("strategic_planning", strategic_planning)
builder.add_node("execute_coordination", execute_coordination)
builder.add_node("monitor_and_adapt", monitor_and_adapt)

# Define the orchestration flow
builder.add_edge("fetch_system_state", "assess_coordination_need")
builder.add_edge("assess_coordination_need", "strategic_planning")
builder.add_edge("strategic_planning", "execute_coordination")
builder.add_edge("execute_coordination", "monitor_and_adapt")
builder.add_edge("monitor_and_adapt", END)
builder.set_entry_point("fetch_system_state")

graph = builder.compile()
