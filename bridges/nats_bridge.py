
import asyncio, os, json, time, logging
from prometheus_client import start_http_server, Counter, Histogram
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from graphs.sensor_graph import graph as sensor_graph
from graphs.control_graph import graph as control_graph
from graphs.orchestrator_graph import graph as orchestrator_graph
from graphs.maintenance_graph import graph as maintenance_graph
from common.logging import setup_json_logging

log = setup_json_logging()
RUNS = Counter("bridge_graph_runs_total","Runs executed by bridge",["graph"])
LAT = Histogram("bridge_run_seconds","Bridge run duration",["graph"])

async def run_graph_async(g, payload, name):
    """Run LangGraph asynchronously with proper error handling"""
    t0 = time.time()
    try:
        # Try async invocation first (for graphs with async nodes)
        try:
            result = await g.ainvoke(payload or {})
            log.debug(f"Successfully executed {name} asynchronously")
        except AttributeError:
            # Fallback to sync invocation if async not supported
            result = g.invoke(payload or {})
            log.debug(f"Successfully executed {name} synchronously")
        
        RUNS.labels(name).inc()
        LAT.labels(name).observe(time.time() - t0)
        return result
        
    except Exception as e:
        log.error(f"Graph execution failed for {name}: {e}")
        LAT.labels(name).observe(time.time() - t0)
        # Don't re-raise - we want the bridge to keep running even if one graph fails
        return {"error": str(e), "graph": name, "timestamp": time.time()}

async def main():
    """Main bridge loop with enhanced error handling"""
    # Start metrics server
    start_http_server(int(os.getenv("BRIDGE_METRICS_PORT","9005")))
    log.info(f"Started metrics server on port {os.getenv('BRIDGE_METRICS_PORT','9005')}")
    
    # Connect to NATS
    servers = os.getenv("NATS_URL","nats://nats:4222")
    nc = NATS()
    
    try:
        await nc.connect(servers=servers)
        log.info(f"Connected to NATS at {servers}")
    except Exception as e:
        log.error(f"Failed to connect to NATS: {e}")
        return
    
    # Setup JetStream
    js = nc.jetstream()
    try: 
        await js.add_stream(StreamConfig(name="TRIG", subjects=["dc.trigger.*"]))
        log.info("Created/verified TRIG stream")
    except Exception as e:
        log.warning(f"Stream setup warning: {e}")

    async def handle_msg(msg):
        """Handle incoming trigger messages"""
        subj = msg.subject
        payload = {}
        
        try: 
            payload = json.loads(msg.data.decode())
            log.debug(f"Received message on {subj} with payload keys: {list(payload.keys())}")
        except json.JSONDecodeError as e:
            log.warning(f"Failed to decode JSON payload for {subj}: {e}")
        except Exception as e:
            log.error(f"Unexpected error decoding message: {e}")
            
        # Route messages to appropriate graphs
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

    # Subscribe to all trigger messages
    try:
        await nc.subscribe("dc.trigger.*", cb=handle_msg)
        log.info("NATS bridge subscribed to dc.trigger.*")
    except Exception as e:
        log.error(f"Failed to subscribe to triggers: {e}")
        return
    
    # Keep the bridge running
    try:
        while True:
            await asyncio.sleep(3600)  # Wake up every hour for maintenance
            log.debug("Bridge heartbeat - still running")
    except KeyboardInterrupt:
        log.info("Bridge shutdown requested")
    except Exception as e:
        log.error(f"Bridge main loop error: {e}")
    finally:
        if nc and nc.is_connected:
            await nc.close()
            log.info("NATS connection closed")

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bridge stopped by user")
    except Exception as e:
        log.error(f"Bridge startup failed: {e}")
        exit(1)