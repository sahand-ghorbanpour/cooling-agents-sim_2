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

async def maybe_update_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
    """Maybe update maintenance schedule based on input"""
    gname="maintenance_graph"; nname="maybe_update_schedule"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        interval_h = state.get("interval_hours")
        if interval_h is not None:
            set_config("maintenance_interval_hours", float(interval_h))
            await nats_publish("dc.config.maintenance", {"interval_hours": float(interval_h)}, agent="maintenance")
            log.info(f"Updated maintenance interval to {interval_h} hours")
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def run_checks(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run maintenance checks and generate report"""
    gname="maintenance_graph"; nname="run_checks"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        # Perform basic health checks
        current_time = time.time()
        
        # Simulate maintenance checks
        checks_performed = [
            "cooling_tower_fan_inspection",
            "cabinet_filter_status", 
            "water_quality_test",
            "valve_operation_test",
            "temperature_sensor_calibration"
        ]
        
        # Generate maintenance report
        report = {
            "timestamp": current_time,
            "checks_performed": checks_performed,
            "fans_ok": True,
            "filters_clean": True,
            "water_quality": "good",
            "valve_functionality": "optimal",
            "sensor_accuracy": "within_tolerance",
            "recommendations": [],
            "next_maintenance": current_time + (24 * 3600)  # 24 hours from now
        }
        
        # Add some realistic maintenance findings occasionally
        import random
        if random.random() < 0.1:  # 10% chance of finding issues
            report["filters_clean"] = False
            report["recommendations"].append("Replace cabinet air filters in 2 weeks")
        
        if random.random() < 0.05:  # 5% chance of sensor drift
            report["sensor_accuracy"] = "minor_drift_detected" 
            report["recommendations"].append("Schedule sensor recalibration")
        
        # Determine overall status
        issues_found = not all([
            report["fans_ok"],
            report["filters_clean"], 
            report["water_quality"] == "good",
            report["valve_functionality"] == "optimal",
            report["sensor_accuracy"] == "within_tolerance"
        ])
        
        report["status"] = "needs_attention" if issues_found else "all_clear"
        
        # Publish maintenance report
        await nats_publish("dc.maintenance.report", {"report": report}, agent="maintenance")
        
        # Also publish to general telemetry for monitoring
        maintenance_summary = {
            "type": "maintenance.summary",
            "status": report["status"],
            "checks_count": len(checks_performed),
            "issues_count": len(report["recommendations"]),
            "timestamp": current_time
        }
        await nats_publish("dc.telemetry.maintenance", maintenance_summary, agent="maintenance")
        
        state["report"] = report
        
        log.info(f"Maintenance checks completed: status={report['status']}, "
                f"recommendations={len(report['recommendations'])}")
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

async def schedule_next_maintenance(state: Dict[str, Any]) -> Dict[str, Any]:
    """Schedule next maintenance based on findings"""
    gname="maintenance_graph"; nname="schedule_next_maintenance"
    t0=time.time(); node_runs_total.labels(gname,nname).inc()
    
    with tracer.start_as_current_span(nname):
        report = state.get("report", {})
        current_interval = float(state.get("interval_hours", 12.0))
        
        # Adjust next maintenance interval based on findings
        if report.get("status") == "needs_attention":
            # Schedule sooner if issues found
            next_interval = max(6.0, current_interval * 0.75)  # Reduce by 25%, minimum 6 hours
            log.warning(f"Issues found - reducing maintenance interval to {next_interval} hours")
        elif len(report.get("recommendations", [])) == 0:
            # Extend interval if all systems are perfect
            next_interval = min(24.0, current_interval * 1.1)  # Increase by 10%, maximum 24 hours
            log.info(f"All systems optimal - extending maintenance interval to {next_interval} hours")
        else:
            # Keep current interval
            next_interval = current_interval
        
        # Update configuration
        set_config("maintenance_interval_hours", next_interval)
        
        # Publish scheduling update
        scheduling_update = {
            "next_interval_hours": next_interval,
            "reason": report.get("status", "routine"),
            "scheduled_time": time.time() + (next_interval * 3600),
            "timestamp": time.time()
        }
        
        await nats_publish("dc.maintenance.schedule", scheduling_update, agent="maintenance")
        
        state["next_interval"] = next_interval
        state["scheduling_update"] = scheduling_update
        
        node_duration_seconds.labels(gname,nname).observe(time.time()-t0)
        return state

# Build the maintenance graph
builder = StateGraph(dict)
builder.add_node("maybe_update_schedule", maybe_update_schedule)
builder.add_node("run_checks", run_checks)
builder.add_node("schedule_next_maintenance", schedule_next_maintenance)

# Define the maintenance flow
builder.add_edge("maybe_update_schedule", "run_checks")
builder.add_edge("run_checks", "schedule_next_maintenance")
builder.add_edge("schedule_next_maintenance", END)
builder.set_entry_point("maybe_update_schedule")

graph = builder.compile()
