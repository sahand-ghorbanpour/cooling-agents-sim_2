
import asyncio, os, time, logging
from nats.aio.client import Client as NATS
from prometheus_client import start_http_server
from common.config import get_config
from common.logging import setup_json_logging

log = setup_json_logging()
start_http_server(int(os.getenv("SCHED_METRICS_PORT","9006")))

async def main():
    nc = NATS(); await nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
    while True:
        sensor_ms = int(get_config("sensor_interval_ms", 1000))
        control_ms = int(get_config("control_interval_ms", 1500))
        maint_h   = float(get_config("maintenance_interval_hours", 12.0))
        await nc.publish("dc.trigger.sensor", b"{}")
        await asyncio.sleep(sensor_ms/1000.0)
        await nc.publish("dc.trigger.control", b"{}")
        # fire orchestrator roughly every maint interval
        if int(time.time()) % int(max(1, maint_h*3600)) < sensor_ms/1000.0:
            await nc.publish("dc.trigger.orchestrator", b"{}")
        await asyncio.sleep(0.05)

if __name__=="__main__":
    asyncio.run(main())
