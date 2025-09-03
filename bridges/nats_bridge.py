import asyncio, os, json, time, logging
from prometheus_client import start_http_server, Counter, Histogram
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from graphs.sensor_graph import graph as sensor_graph
from graphs.control_graph import graph as control_graph
from graphs.orchestrator_graph import graph as orchestrator_graph
from graphs.maintenance_graph import graph as maintenance_graph
from common.logging import setup_json_logging
from common.config import set_latest_state

log = setup_json_logging()
RUNS = Counter("bridge_graph_runs_total","Runs executed by bridge",["graph"])
LAT = Histogram("bridge_run_seconds","Bridge run duration",["graph"])

# BRIDGE-MANAGED PERSISTENT DATA
latest_simulation_data = None
simulation_health = {"status": "unknown", "last_update": 0}

async def run_graph_async(g, payload, name):
    """Run LangGraph asynchronously with injected data"""
    t0 = time.time()
    try:
        # INJECT SIMULATION DATA INTO SENSOR GRAPH
        if name == "sensor_graph" and latest_simulation_data:
            payload["injected_simulation_data"] = latest_simulation_data
            payload["simulation_health"] = simulation_health
        
        # Try async invocation first
        try:
            result = await g.ainvoke(payload or {})
            log.debug(f"Successfully executed {name} asynchronously")
        except AttributeError:
            result = g.invoke(payload or {})
            log.debug(f"Successfully executed {name} synchronously")
        
        RUNS.labels(name).inc()
        LAT.labels(name).observe(time.time() - t0)
        return result
        
    except Exception as e:
        log.error(f"Graph execution failed for {name}: {e}")
        LAT.labels(name).observe(time.time() - t0)
        return {"error": str(e), "graph": name, "timestamp": time.time()}

async def handle_simulation_data(msg):
    """Handle simulation data from sensor-agent"""
    global latest_simulation_data, simulation_health
    
    try:
        payload = json.loads(msg.data.decode())
        latest_simulation_data = payload
        simulation_health = {
            "status": "healthy",
            "last_update": time.time()
        }
        
        # Also store in Redis for other services
        set_latest_state(payload.get("data", {}))
        
        log.debug("Updated simulation data from sensor-agent")
        
    except Exception as e:
        log.error(f"Error processing simulation data: {e}")

async def main():
    """Main bridge loop with simulation data management"""
    global latest_simulation_data, simulation_health
    
    start_http_server(int(os.getenv("BRIDGE_METRICS_PORT","9005")))
    
    nc = NATS()
    await nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
    
    js = nc.jetstream()
    try: 
        await js.add_stream(StreamConfig(name="TRIG", subjects=["dc.trigger.*"]))
    except Exception as e:
        log.warning(f"Stream setup warning: {e}")

    # SUBSCRIBE TO SIMULATION DATA FROM SENSOR-AGENT
    await nc.subscribe("simulation.state.raw", cb=handle_simulation_data)
    await nc.subscribe("dc.sensor.telemetry.raw", cb=handle_simulation_data)
    log.info("Bridge subscribed to simulation data topics")

    async def handle_msg(msg):
        """Handle trigger messages"""
        subj = msg.subject
        payload = {}
        
        try: 
            payload = json.loads(msg.data.decode())
        except json.JSONDecodeError as e:
            log.warning(f"Failed to decode JSON payload for {subj}: {e}")
            
        # Route messages to graphs with data injection
        try:
            if subj == "dc.trigger.sensor":
                await run_graph_async(sensor_graph, payload, "sensor_graph")
            elif subj == "dc.trigger.control":
                await run_graph_async(control_graph, payload, "control_graph")
            elif subj == "dc.trigger.orchestrator":
                await run_graph_async(orchestrator_graph, payload, "orchestrator_graph")
            elif subj == "dc.trigger.maintenance":
                await run_graph_async(maintenance_graph, payload, "maintenance_graph")
            else:
                log.warning(f"Unknown trigger subject: {subj}")
                
        except Exception as e:
            log.error(f"Message handling failed for {subj}: {e}")

    await nc.subscribe("dc.trigger.*", cb=handle_msg)
    log.info("NATS bridge ready - managing persistent simulation data")
    
    # Keep running and monitor data freshness
    try:
        while True:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            # Check if simulation data is stale
            if simulation_health["last_update"] > 0:
                age = time.time() - simulation_health["last_update"]
                if age > 60:  # More than 1 minute old
                    simulation_health["status"] = "stale"
                    log.warning(f"Simulation data is stale ({age:.1f}s old)")
                    
    except KeyboardInterrupt:
        log.info("Bridge shutdown requested")
    finally:
        if nc and nc.is_connected:
            await nc.close()

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bridge stopped by user")
    except Exception as e:
        log.error(f"Bridge startup failed: {e}")
        exit(1)