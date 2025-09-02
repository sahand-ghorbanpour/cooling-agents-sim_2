
import asyncio, json, os
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig, RetentionPolicy

ROUTING = {
    "state_out": "dc.telemetry.state",
    "actions_out": "dc.control.actions",
    "orch_targets": "dc.orch.targets",
    "triggers": {
        "sensor": "dc.trigger.sensor",
        "control": "dc.trigger.control",
        "orchestrator": "dc.trigger.orchestrator"
    }
}

async def get_js():
    servers=os.getenv("NATS_URL","nats://nats:4222")
    nc=NATS(); await nc.connect(servers=servers); js=nc.jetstream()
    # Ensure streams
    for name, subjects in {
        "DC": ["dc.*"],
        "TRIG": ["dc.trigger.*"]
    }.items():
        try: await js.add_stream(StreamConfig(name=name, subjects=subjects, retention=RetentionPolicy.WorkQueue))
        except Exception: pass
    return nc, js

async def publish(subject, payload:dict, agent="sys", direction="out"):
    from common.metrics import nats_messages_total, nats_msg_bytes
    nc, js = await get_js()
    data = json.dumps(payload).encode()
    await js.publish(subject, data)
    nats_messages_total.labels(agent, subject, direction).inc()
    nats_msg_bytes.labels(agent, subject, direction).inc(len(data))
    await nc.close()
