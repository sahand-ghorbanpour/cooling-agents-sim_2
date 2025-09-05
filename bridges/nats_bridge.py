# bridges/nats_bridge.py - Optimized for streamlined data flow

import asyncio, os, json, time, logging
from prometheus_client import start_http_server, Counter, Histogram
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from graphs.sensor_graph import graph as sensor_graph, run_optimized_sensor_processing
from graphs.control_graph import graph as control_graph
from graphs.orchestrator_graph import graph as orchestrator_graph
from graphs.maintenance_graph import graph as maintenance_graph
from common.logging import setup_json_logging
from common.nats_utils import get_nats_connection

log = setup_json_logging()
RUNS = Counter("bridge_graph_runs_total","Runs executed by bridge",["graph", "mode"])
LAT = Histogram("bridge_run_seconds","Bridge run duration",["graph", "mode"])
ERRORS = Counter("bridge_errors_total", "Bridge execution errors", ["graph", "error_type"])

class OptimizedNATSBridge:
    def __init__(self):
        self.nc = None
        self.use_direct_processing = True  # Use optimized direct processing when possible
        self.graph_health = {
            "sensor_graph": {"last_success": 0, "error_count": 0},
            "control_graph": {"last_success": 0, "error_count": 0},
            "orchestrator_graph": {"last_success": 0, "error_count": 0},
            "maintenance_graph": {"last_success": 0, "error_count": 0}
        }
        
    async def initialize(self):
        """Initialize NATS connection and streams"""
        self.nc = await get_nats_connection()
        
        js = self.nc.jetstream()
        try: 
            await js.add_stream(StreamConfig(
                name="TRIG", 
                subjects=["dc.trigger.*"],
                max_age=300  # 5 minute retention for triggers
            ))
        except Exception as e:
            log.warning(f"Stream setup warning: {e}")
        
        log.info("Optimized NATS bridge initialized")
    
    async def run_graph_optimized(self, graph, payload, name):
        """Run graph with optimization and error handling"""
        mode = "direct" if name == "sensor_graph" and self.use_direct_processing else "graph"
        t0 = time.time()
        
        try:
            # Use direct processing for sensor graph when possible
            if name == "sensor_graph" and self.use_direct_processing:
                await run_optimized_sensor_processing()
                result = {"status": "success", "mode": "direct_processing"}
            else:
                # Use LangGraph for complex coordination
                try:
                    result = await graph.ainvoke(payload or {})
                except AttributeError:
                    result = graph.invoke(payload or {})
            
            # Update health tracking
            self.graph_health[name]["last_success"] = time.time()
            self.graph_health[name]["error_count"] = 0
            
            RUNS.labels(graph=name, mode=mode).inc()
            LAT.labels(graph=name, mode=mode).observe(time.time() - t0)
            
            log.debug(f"Successfully executed {name} in {mode} mode")
            return result
            
        except Exception as e:
            # Update error tracking
            self.graph_health[name]["error_count"] += 1
            ERRORS.labels(graph=name, error_type=type(e).__name__).inc()
            
            log.error(f"Graph execution failed for {name}: {e}")
            
            # Switch to fallback mode for sensor graph
            if name == "sensor_graph" and mode == "direct":
                log.info("Switching sensor graph to LangGraph mode due to direct processing error")
                self.use_direct_processing = False
                return await self.run_graph_optimized(graph, payload, name)
            
            LAT.labels(graph=name, mode=mode).observe(time.time() - t0)
            return {"error": str(e), "graph": name, "timestamp": time.time()}
    
    async def handle_trigger_message(self, msg):
        """Handle trigger messages with optimized routing"""
        subj = msg.subject
        payload = {}
        
        try: 
            payload = json.loads(msg.data.decode())
        except json.JSONDecodeError as e:
            log.warning(f"Failed to decode JSON payload for {subj}: {e}")
            payload = {}
        
        # Add execution metadata
        payload["bridge_timestamp"] = time.time()
        payload["subject"] = subj
        
        # Route to appropriate graph with error handling
        try:
            if subj == "dc.trigger.sensor":
                await self.run_graph_optimized(sensor_graph, payload, "sensor_graph")
            elif subj == "dc.trigger.control":
                await self.run_graph_optimized(control_graph, payload, "control_graph")
            elif subj == "dc.trigger.orchestrator":
                await self.run_graph_optimized(orchestrator_graph, payload, "orchestrator_graph")
            elif subj == "dc.trigger.maintenance":
                await self.run_graph_optimized(maintenance_graph, payload, "maintenance_graph")
            else:
                log.warning(f"Unknown trigger subject: {subj}")
                
        except Exception as e:
            log.error(f"Message handling failed for {subj}: {e}")
            ERRORS.labels(graph=subj.split(".")[-1], error_type="routing_error").inc()
    
    async def health_monitor(self):
        """Monitor graph health and adapt processing modes"""
        log.info("Starting health monitor")
        
        while True:
            try:
                current_time = time.time()
                
                for graph_name, health in self.graph_health.items():
                    # Check if graph is unhealthy
                    time_since_success = current_time - health["last_success"]
                    error_count = health["error_count"]
                    
                    if time_since_success > 300 and error_count > 5:  # 5 minutes, 5+ errors
                        log.warning(f"{graph_name} appears unhealthy: {error_count} errors, "
                                  f"last success {time_since_success:.1f}s ago")
                    
                    # Auto-recovery for sensor graph
                    if (graph_name == "sensor_graph" and 
                        not self.use_direct_processing and 
                        error_count == 0 and 
                        time_since_success < 60):
                        log.info("Re-enabling direct processing for sensor graph")
                        self.use_direct_processing = True
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                log.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def run(self):
        """Main bridge loop"""
        await self.initialize()
        
        # Subscribe to trigger messages
        await self.nc.subscribe("dc.trigger.*", cb=self.handle_trigger_message)
        log.info("NATS bridge ready - optimized for streamlined data flow")
        
        # Start health monitor as background task
        health_task = asyncio.create_task(self.health_monitor())
        
        try:
            # Keep running
            await health_task
        except KeyboardInterrupt:
            health_task.cancel()
            log.info("Bridge shutdown requested")
        finally:
            if self.nc and self.nc.is_connected:
                await self.nc.close()

async def main():
    """Main bridge entry point"""
    start_http_server(int(os.getenv("BRIDGE_METRICS_PORT","9005")))
    
    bridge = OptimizedNATSBridge()
    await bridge.run()

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bridge stopped by user")
    except Exception as e:
        log.error(f"Bridge startup failed: {e}")
        exit(1)