# bridges/scheduler.py - Optimized to reduce race conditions and timing dependencies

import asyncio, os, time, logging
from nats.aio.client import Client as NATS
from prometheus_client import start_http_server, Counter, Gauge
from common.config import get_config, get_latest_state
from common.logging import setup_json_logging
from common.nats_utils import get_nats_connection

log = setup_json_logging()
start_http_server(int(os.getenv("SCHED_METRICS_PORT","9006")))

# Metrics
scheduler_triggers = Counter('scheduler_triggers_total', 'Scheduler triggers sent', ['graph_type'])
scheduler_interval = Gauge('scheduler_interval_seconds', 'Current scheduling interval', ['graph_type'])
data_freshness = Gauge('data_freshness_seconds', 'Age of latest sensor data')

class OptimizedScheduler:
    def __init__(self):
        self.nc = None
        self.last_trigger_times = {}
        self.adaptive_intervals = {
            "sensor": 1.0,      # Fast sensor processing
            "control": 2.0,     # LLM control (slower)
            "orchestrator": 60.0,  # Orchestration
            "maintenance": 3600.0  # Maintenance
        }
        self.data_driven_mode = True  # Trigger based on data availability, not just time
        
    async def initialize(self):
        """Initialize NATS connection"""
        self.nc = await get_nats_connection()
        log.info("Optimized scheduler initialized")
    
    async def should_trigger_sensor(self) -> bool:
        """Determine if sensor graph should be triggered - data-driven"""
        if not self.data_driven_mode:
            return self._time_based_trigger("sensor")
        
        # Check data freshness - trigger only if data is stale
        latest_data = get_latest_state({})
        if not latest_data:
            return True  # No data, trigger processing
        
        data_age = time.time() - latest_data.get("ts", 0)
        data_freshness.set(data_age)
        
        # Trigger if data is older than interval or if we haven't triggered recently
        last_trigger = self.last_trigger_times.get("sensor", 0)
        time_since_trigger = time.time() - last_trigger
        
        should_trigger = (
            data_age > self.adaptive_intervals["sensor"] * 1.5 or  # Data is stale
            time_since_trigger > self.adaptive_intervals["sensor"] * 2  # Force trigger
        )
        
        return should_trigger
    
    async def should_trigger_control(self) -> bool:
        """Determine if control graph should be triggered - state-driven"""
        # Control triggers based on:
        # 1. Time interval
        # 2. Temperature deviation
        # 3. System changes
        
        if not self._time_based_trigger("control"):
            return False
        
        latest_data = get_latest_state({})
        if not latest_data:
            return False
        
        temps = latest_data.get("temps", {})
        if not temps:
            return False
        
        # Check if control is needed based on temperature deviation
        cabinet_temps = [v for k, v in temps.items() if k.startswith("cabinet_")]
        if cabinet_temps:
            avg_temp = sum(cabinet_temps) / len(cabinet_temps)
            target_temp = float(get_config("target_temp_c", 24.0))
            temp_error = abs(avg_temp - target_temp)
            
            # Trigger more frequently if temperature error is high
            if temp_error > 2.0:
                self.adaptive_intervals["control"] = 1.0  # Every second for critical
            elif temp_error > 1.0:
                self.adaptive_intervals["control"] = 1.5  # Fast for high error
            else:
                self.adaptive_intervals["control"] = 3.0  # Normal for stable
            
            scheduler_interval.labels(graph_type="control").set(self.adaptive_intervals["control"])
        
        return True
    
    async def should_trigger_orchestrator(self) -> bool:
        """Determine if orchestrator should be triggered - event-driven"""
        # Orchestrator triggers based on:
        # 1. System health issues
        # 2. Performance degradation  
        # 3. Time-based backup
        
        if not self._time_based_trigger("orchestrator"):
            return False
        
        # Check system health indicators
        latest_data = get_latest_state({})
        if latest_data:
            # Check for issues that require orchestration
            meta = latest_data.get("meta", {})
            energy = meta.get("energy_kw", 0)
            
            # Trigger if high energy consumption
            if energy > 150.0:
                log.info("Triggering orchestrator due to high energy consumption")
                return True
            
            # Check temperature spread
            temps = latest_data.get("temps", {})
            cabinet_temps = [v for k, v in temps.items() if k.startswith("cabinet_")]
            if len(cabinet_temps) > 1:
                temp_spread = max(cabinet_temps) - min(cabinet_temps)
                if temp_spread > 4.0:
                    log.info("Triggering orchestrator due to temperature imbalance")
                    return True
        
        # Time-based trigger as fallback
        last_trigger = self.last_trigger_times.get("orchestrator", 0)
        return time.time() - last_trigger > self.adaptive_intervals["orchestrator"]
    
    def _time_based_trigger(self, graph_type: str) -> bool:
        """Check if enough time has passed for time-based trigger"""
        last_trigger = self.last_trigger_times.get(graph_type, 0)
        interval = self.adaptive_intervals[graph_type]
        return time.time() - last_trigger >= interval
    
    async def trigger_graph(self, graph_type: str, payload: dict = None):
        """Trigger a specific graph with payload"""
        try:
            topic = f"dc.trigger.{graph_type}"
            message = payload or {}
            
            await self.nc.publish(topic, str(message).encode() if message else b"{}")
            
            self.last_trigger_times[graph_type] = time.time()
            scheduler_triggers.labels(graph_type=graph_type).inc()
            
            log.debug(f"Triggered {graph_type} graph")
            
        except Exception as e:
            log.error(f"Failed to trigger {graph_type}: {e}")
    
    async def run_adaptive_scheduling(self):
        """Main scheduling loop with adaptive intervals"""
        log.info("Starting adaptive scheduling loop")
        
        while True:
            try:
                # Check each graph type independently
                if await self.should_trigger_sensor():
                    await self.trigger_graph("sensor")
                
                if await self.should_trigger_control():
                    await self.trigger_graph("control")
                
                if await self.should_trigger_orchestrator():
                    await self.trigger_graph("orchestrator")
                
                # Maintenance is always time-based
                if self._time_based_trigger("maintenance"):
                    maintenance_interval = float(get_config("maintenance_interval_hours", 12.0)) * 3600
                    self.adaptive_intervals["maintenance"] = maintenance_interval
                    await self.trigger_graph("maintenance")
                
                # Update interval metrics
                for graph_type, interval in self.adaptive_intervals.items():
                    scheduler_interval.labels(graph_type=graph_type).set(interval)
                
                # Sleep for minimum interval to avoid busy waiting
                await asyncio.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                log.error(f"Scheduling loop error: {e}")
                await asyncio.sleep(1.0)

async def main():
    """Main scheduler entry point"""
    scheduler = OptimizedScheduler()
    await scheduler.initialize()
    
    # Start adaptive scheduling
    await scheduler.run_adaptive_scheduling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user")
    except Exception as e:
        log.error(f"Scheduler startup failed: {e}")
        exit(1)