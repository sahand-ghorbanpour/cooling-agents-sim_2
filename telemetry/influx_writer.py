
import asyncio, os, json
from influxdb_client import InfluxDBClient, Point, WriteOptions
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig

async def main():
    url = os.getenv("INFLUX_URL","http://influxdb:8086")
    token = os.getenv("INFLUX_TOKEN","dev-token")
    org = os.getenv("INFLUX_ORG","dc")
    bucket = os.getenv("INFLUX_BUCKET","telemetry")

    client = InfluxDBClient(url=url, token=token, org=org)
    write_api = client.write_api(write_options=WriteOptions(batch_size=100, flush_interval=1000))

    nc = NATS()
    await nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
    js = nc.jetstream()
    try:
        await js.add_stream(StreamConfig(name="TELE", subjects=["dc.telemetry.state"]))
    except Exception:
        pass

    async def cb(msg):
        try:
            payload = json.loads(msg.data.decode())
            data = payload.get("data", {})
            temps = data.get("temps", {})
            p = Point("cabinet_temps")
            for k,v in temps.items():
                p.field(k, float(v))
            write_api.write(bucket=bucket, org=org, record=p)
        except Exception:
            pass

    await nc.subscribe("dc.telemetry.state", cb=cb)
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
