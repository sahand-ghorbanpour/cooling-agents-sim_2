
import asyncio, os, json, time, logging
from prometheus_client import start_http_server, Counter, Histogram
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig
from graphs.sensor_graph import graph as sensor_graph
from graphs.control_graph import graph as control_graph
from graphs.orchestrator_graph import graph as orchestrator_graph
from common.logging import setup_json_logging

log = setup_json_logging()
RUNS = Counter("bridge_graph_runs_total","Runs executed by bridge",["graph"])
LAT = Histogram("bridge_run_seconds","Bridge run duration",["graph"])

def run_graph_sync(g, payload, name):
    t0=time.time(); g.invoke(payload or {})
    RUNS.labels(name).inc(); LAT.labels(name).observe(time.time()-t0)

async def main():
    start_http_server(int(os.getenv("BRIDGE_METRICS_PORT","9005")))
    servers = os.getenv("NATS_URL","nats://nats:4222")
    nc = NATS(); await nc.connect(servers=servers); js = nc.jetstream()
    try: await js.add_stream(StreamConfig(name="TRIG", subjects=["dc.trigger.*"]))
    except Exception: pass

    async def handle_msg(msg):
        subj = msg.subject
        payload = {}
        try: payload=json.loads(msg.data.decode())
        except Exception: pass
        if subj=="dc.trigger.sensor":
            run_graph_sync(sensor_graph, payload, "sensor_graph")
        elif subj=="dc.trigger.control":
            run_graph_sync(control_graph, payload, "control_graph")
        elif subj=="dc.trigger.orchestrator":
            run_graph_sync(orchestrator_graph, payload, "orchestrator_graph")

    await nc.subscribe("dc.trigger.*", cb=handle_msg)
    log.info("NATS bridge subscribed to dc.trigger.*")
    while True:
        await asyncio.sleep(3600)

if __name__=="__main__":
    asyncio.run(main())
