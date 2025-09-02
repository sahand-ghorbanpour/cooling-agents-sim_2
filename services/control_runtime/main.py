
import asyncio, json, os, logging, time
from nats.aio.client import Client as NATS
from prometheus_client import start_http_server, Counter
from common.logging import setup_json_logging
from common.metrics import control_action

log = setup_json_logging()
start_http_server(int(os.getenv("METRICS_PORT","9010")))

async def main():
    nc = NATS(); await nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
    async def cb(msg):
        try:
            payload=json.loads(msg.data.decode())
            actions=payload.get("data",{})
            # Apply clamps and "actuate"
            for k,v in actions.items():
                pct=max(0.0,min(100.0,float(v)))
                control_action.labels(target=k).set(pct)
            log.info(f"Actuated actions: {actions}")
        except Exception as e:
            log.error(f"actuation error: {e}")
    await nc.subscribe("dc.control.actions", cb=cb)
    while True: await asyncio.sleep(1)

if __name__=="__main__":
    asyncio.run(main())
