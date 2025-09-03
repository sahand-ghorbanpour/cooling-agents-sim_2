import asyncio, json, os, logging, time
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from nats.aio.client import Client as NATS
from common.logging import setup_json_logging
from common.otel import init_tracer
from common.config import get_config, set_config

log = setup_json_logging()
app = FastAPI(title="Sensor Gateway - PLC/Modbus Interface")
tracer = init_tracer("sensor-gateway")
start_http_server(int(os.getenv("METRICS_PORT","9011")))

# Prometheus metrics
sensor_reads_total = Counter('sensor_reads_total', 'Total sensor readings', ['sensor_type', 'cabinet'])
sensor_read_duration = Histogram('sensor_read_duration_seconds', 'Time to read sensors')
sensor_value = Gauge('sensor_raw_value', 'Raw sensor values', ['sensor_id', 'sensor_type'])
communication_errors = Counter('sensor_communication_errors_total', 'Communication errors', ['protocol', 'device'])

# Sensor configuration
class SensorConfig(BaseModel):
    sensor_id: str
    sensor_type: str  # temperature, pressure, flow, power
    cabinet_id: str
    protocol: str  # modbus_tcp, modbus_rtu, opcua, snmp
    address: str
    register: int
    scale_factor: float = 1.0
    offset: float = 0.0
    unit: str = ""
    enabled: bool = True

class SensorReading(BaseModel):
    sensor_id: str
    value: float
    unit: str
    timestamp: float
    quality: str = "good"  # good, uncertain, bad

class SensorGateway:
    def __init__(self):
        self.nc = None
        self.sensors = {}
        self.last_readings = {}
        self.communication_stats = {}
        self._load_sensor_config()
        
    def _load_sensor_config(self):
        """Load sensor configuration from file or environment"""
        # Default sensor configuration for 5-cabinet system
        default_sensors = []
        
        for cabinet_num in range(1, 6):
            cabinet_id = f"cabinet_{cabinet_num}"
            base_addr = f"192.168.1.{100 + cabinet_num}"
            
            # Temperature sensors (3 per cabinet - boundary temperatures)
            for temp_num in range(1, 4):
                default_sensors.append(SensorConfig(
                    sensor_id=f"{cabinet_id}_temp_{temp_num}",
                    sensor_type="temperature",
                    cabinet_id=cabinet_id,
                    protocol="modbus_tcp",
                    address=base_addr,
                    register=40000 + temp_num,  # Holding registers
                    scale_factor=0.1,  # Convert to Celsius
                    offset=-273.15,    # Kelvin to Celsius
                    unit="celsius"
                ))
            
            # Power sensors (3 per cabinet - blade power)
            for power_num in range(1, 4):
                default_sensors.append(SensorConfig(
                    sensor_id=f"{cabinet_id}_power_{power_num}",
                    sensor_type="power",
                    cabinet_id=cabinet_id,
                    protocol="modbus_tcp", 
                    address=base_addr,
                    register=40010 + power_num,
                    scale_factor=1000.0,  # Convert to Watts
                    unit="watts"
                ))
            
            # Flow sensors
            default_sensors.append(SensorConfig(
                sensor_id=f"{cabinet_id}_flow_rate",
                sensor_type="flow",
                cabinet_id=cabinet_id,
                protocol="modbus_tcp",
                address=base_addr,
                register=40020,
                scale_factor=0.01,  # Convert to L/s
                unit="l_per_s"
            ))
            
            # Pressure sensors
            default_sensors.append(SensorConfig(
                sensor_id=f"{cabinet_id}_pressure",
                sensor_type="pressure",
                cabinet_id=cabinet_id,
                protocol="modbus_tcp",
                address=base_addr,
                register=40025,
                scale_factor=0.01,  # Convert to kPa
                unit="kpa"
            ))
        
        # Cooling tower sensors
        ct_address = "192.168.1.200"
        for fan_num in range(1, 5):
            default_sensors.extend([
                SensorConfig(
                    sensor_id=f"cooling_tower_fan_{fan_num}_power",
                    sensor_type="power",
                    cabinet_id="cooling_tower",
                    protocol="modbus_tcp",
                    address=ct_address,
                    register=40000 + fan_num,
                    scale_factor=1000.0,
                    unit="watts"
                ),
                SensorConfig(
                    sensor_id=f"cooling_tower_fan_{fan_num}_speed",
                    sensor_type="speed",
                    cabinet_id="cooling_tower", 
                    protocol="modbus_tcp",
                    address=ct_address,
                    register=40010 + fan_num,
                    scale_factor=1.0,
                    unit="rpm"
                )
            ])
        
        # Water temperature sensors
        default_sensors.extend([
            SensorConfig(
                sensor_id="cooling_tower_water_temp_supply",
                sensor_type="temperature",
                cabinet_id="cooling_tower",
                protocol="modbus_tcp",
                address=ct_address,
                register=40030,
                scale_factor=0.1,
                unit="celsius"
            ),
            SensorConfig(
                sensor_id="cooling_tower_water_temp_return", 
                sensor_type="temperature",
                cabinet_id="cooling_tower",
                protocol="modbus_tcp",
                address=ct_address,
                register=40031,
                scale_factor=0.1,
                unit="celsius"
            ),
            SensorConfig(
                sensor_id="ambient_wet_bulb_temp",
                sensor_type="temperature",
                cabinet_id="ambient",
                protocol="modbus_tcp",
                address=ct_address,
                register=40040,
                scale_factor=0.1,
                unit="celsius"
            )
        ])
        
        # Store sensors by ID
        for sensor in default_sensors:
            self.sensors[sensor.sensor_id] = sensor
            
        log.info(f"Loaded {len(self.sensors)} sensor configurations")
    
    async def initialize(self):
        """Initialize NATS connection and start sensor reading loop"""
        self.nc = NATS()
        await self.nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
        log.info("Sensor gateway initialized")
    
    async def read_all_sensors(self) -> Dict[str, SensorReading]:
        """Read all configured sensors"""
        readings = {}
        
        with sensor_read_duration.time():
            # Group sensors by protocol and address for efficient reading
            protocol_groups = {}
            for sensor in self.sensors.values():
                if not sensor.enabled:
                    continue
                    
                key = f"{sensor.protocol}:{sensor.address}"
                if key not in protocol_groups:
                    protocol_groups[key] = []
                protocol_groups[key].append(sensor)
            
            # Read each protocol group
            for group_key, sensor_list in protocol_groups.items():
                protocol, address = group_key.split(":", 1)
                
                try:
                    group_readings = await self._read_sensor_group(protocol, address, sensor_list)
                    readings.update(group_readings)
                    
                except Exception as e:
                    communication_errors.labels(protocol=protocol, device=address).inc()
                    log.error(f"Failed to read sensor group {group_key}: {e}")
                    
                    # Generate fault readings for this group
                    for sensor in sensor_list:
                        readings[sensor.sensor_id] = SensorReading(
                            sensor_id=sensor.sensor_id,
                            value=0.0,
                            unit=sensor.unit,
                            timestamp=time.time(),
                            quality="bad"
                        )
        
        self.last_readings = readings
        return readings
    
    async def _read_sensor_group(self, protocol: str, address: str, sensors: List[SensorConfig]) -> Dict[str, SensorReading]:
        """Read a group of sensors using the same protocol and address"""
        readings = {}
        
        if protocol == "modbus_tcp":
            readings = await self._read_modbus_tcp(address, sensors)
        elif protocol == "modbus_rtu":
            readings = await self._read_modbus_rtu(address, sensors)
        elif protocol == "opcua":
            readings = await self._read_opcua(address, sensors)
        elif protocol == "snmp":
            readings = await self._read_snmp(address, sensors)
        else:
            # Mock/simulation mode
            readings = await self._read_simulation(address, sensors)
        
        return readings
    
    async def _read_modbus_tcp(self, address: str, sensors: List[SensorConfig]) -> Dict[str, SensorReading]:
        """Read Modbus TCP sensors (placeholder - would use pymodbus in production)"""
        readings = {}
        
        # In production, this would use pymodbus:
        # from pymodbus.client import AsyncModbusTcpClient
        # client = AsyncModbusTcpClient(address)
        
        # For now, simulate realistic readings
        for sensor in sensors:
            raw_value = await self._simulate_sensor_value(sensor)
            scaled_value = (raw_value * sensor.scale_factor) + sensor.offset
            
            readings[sensor.sensor_id] = SensorReading(
                sensor_id=sensor.sensor_id,
                value=scaled_value,
                unit=sensor.unit,
                timestamp=time.time(),
                quality="good"
            )
            
            # Update metrics
            sensor_reads_total.labels(
                sensor_type=sensor.sensor_type,
                cabinet=sensor.cabinet_id
            ).inc()
            sensor_value.labels(
                sensor_id=sensor.sensor_id,
                sensor_type=sensor.sensor_type
            ).set(scaled_value)
        
        return readings
    
    async def _read_modbus_rtu(self, address: str, sensors: List[SensorConfig]) -> Dict[str, SensorReading]:
        """Read Modbus RTU sensors"""
        # Similar to TCP but over serial
        return await self._read_modbus_tcp(address, sensors)  # Placeholder
    
    async def _read_opcua(self, address: str, sensors: List[SensorConfig]) -> Dict[str, SensorReading]:
        """Read OPC UA sensors"""  
        # Would use asyncua library in production
        return await self._read_modbus_tcp(address, sensors)  # Placeholder
    
    async def _read_snmp(self, address: str, sensors: List[SensorConfig]) -> Dict[str, SensorReading]:
        """Read SNMP sensors"""
        # Would use pysnmp library in production
        return await self._read_modbus_tcp(address, sensors)  # Placeholder
    
    async def _read_simulation(self, address: str, sensors: List[SensorConfig]) -> Dict[str, SensorReading]:
        """Generate simulated sensor readings for testing"""
        readings = {}
        
        for sensor in sensors:
            value = await self._simulate_sensor_value(sensor)
            
            readings[sensor.sensor_id] = SensorReading(
                sensor_id=sensor.sensor_id,
                value=value,
                unit=sensor.unit,
                timestamp=time.time(),
                quality="good"
            )
        
        return readings
    
    async def _simulate_sensor_value(self, sensor: SensorConfig) -> float:
        """Generate realistic sensor values for testing"""
        base_values = {
            "temperature": 25.0 + np.random.normal(0, 2.0),  # Around 25°C ± 2°C
            "power": 50000 + np.random.normal(0, 5000),      # Around 50kW ± 5kW  
            "flow": 5.0 + np.random.normal(0, 0.5),          # Around 5 L/s ± 0.5
            "pressure": 100.0 + np.random.normal(0, 5.0),    # Around 100 kPa ± 5
            "speed": 1800 + np.random.normal(0, 100)         # Around 1800 RPM ± 100
        }
        
        base_value = base_values.get(sensor.sensor_type, 0.0)
        
        # Add some cabinet-specific variation
        if "cabinet_1" in sensor.sensor_id:
            base_value += 1.0
        elif "cabinet_5" in sensor.sensor_id:
            base_value -= 1.0
        
        return max(0.0, base_value)
    
    async def publish_telemetry(self, readings: Dict[str, SensorReading]):
        """Publish sensor readings to NATS"""
        # Convert to simplified format for compatibility
        temps = {}
        powers = {}
        flows = {}
        
        for reading in readings.values():
            if reading.quality != "good":
                continue
                
            if "temp" in reading.sensor_id:
                # Map to cabinet format expected by control system
                if "cabinet_1" in reading.sensor_id:
                    temps["cabinet_1"] = reading.value
                elif "cabinet_2" in reading.sensor_id:
                    temps["cabinet_2"] = reading.value
                elif "cabinet_3" in reading.sensor_id:
                    temps["cabinet_3"] = reading.value
                elif "cabinet_4" in reading.sensor_id:
                    temps["cabinet_4"] = reading.value
                elif "cabinet_5" in reading.sensor_id:
                    temps["cabinet_5"] = reading.value
                elif "cooling_tower" in reading.sensor_id:
                    temps["cooling_tower"] = reading.value
            
            elif "power" in reading.sensor_id:
                powers[reading.sensor_id] = reading.value
            elif "flow" in reading.sensor_id:
                flows[reading.sensor_id] = reading.value
        
        # Aggregate multiple temperature sensors per cabinet (take average)
        cabinet_temps = {}
        for i in range(1, 6):
            cabinet_key = f"cabinet_{i}"
            temp_sensors = [r for r in readings.values() 
                          if cabinet_key in r.sensor_id and "temp" in r.sensor_id and r.quality == "good"]
            if temp_sensors:
                avg_temp = sum(r.value for r in temp_sensors) / len(temp_sensors)
                cabinet_temps[cabinet_key] = avg_temp
        
        # Create telemetry payload
        telemetry_payload = {
            "type": "sensor.telemetry",
            "data": {
                "temps": cabinet_temps,
                "powers": powers,
                "flows": flows,
                "raw_readings": {r.sensor_id: r.dict() for r in readings.values()}
            },
            "timestamp": time.time(),
            "source": "sensor_gateway"
        }
        
        # Publish to multiple topics
        await self.nc.publish("dc.sensor.telemetry", json.dumps(telemetry_payload).encode())
        await self.nc.publish("dc.telemetry.state", json.dumps(telemetry_payload).encode())
        
        log.debug(f"Published telemetry: {len(temps)} temperatures, {len(powers)} power readings")
    
    async def run_sensor_loop(self):
        """Main sensor reading loop"""
        log.info("Starting sensor reading loop")
        
        while True:
            try:
                # Read all sensors
                readings = await self.read_all_sensors()
                
                # Publish telemetry
                await self.publish_telemetry(readings)
                
                # Reading frequency from config
                read_interval = float(get_config("sensor_read_interval_ms", 1000)) / 1000.0
                await asyncio.sleep(read_interval)
                
            except Exception as e:
                log.error(f"Sensor loop error: {e}")
                await asyncio.sleep(5.0)

# Global gateway instance
gateway = SensorGateway()

# FastAPI endpoints for MCP tools
@app.get("/healthz")
def health_check():
    return {
        "status": "healthy", 
        "sensors": len(gateway.sensors),
        "last_reading": max([r.timestamp for r in gateway.last_readings.values()]) if gateway.last_readings else 0
    }

@app.get("/sensors")
def list_sensors():
    """List all configured sensors"""
    return [sensor.dict() for sensor in gateway.sensors.values()]

@app.get("/readings")
def get_latest_readings():
    """Get latest sensor readings"""
    return [reading.dict() for reading in gateway.last_readings.values()]

@app.get("/readings/{sensor_id}")
def get_sensor_reading(sensor_id: str):
    """Get specific sensor reading"""
    if sensor_id not in gateway.last_readings:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    return gateway.last_readings[sensor_id].dict()

@app.post("/sensors/{sensor_id}/enable")
def enable_sensor(sensor_id: str, enabled: bool = True):
    """Enable/disable a sensor"""
    if sensor_id not in gateway.sensors:
        raise HTTPException(status_code=404, detail=f"Sensor {sensor_id} not found")
    
    gateway.sensors[sensor_id].enabled = enabled
    return {"sensor_id": sensor_id, "enabled": enabled}

# Main function
async def main():
    await gateway.initialize()
    # Start sensor reading loop as background task
    sensor_task = asyncio.create_task(gateway.run_sensor_loop())
    
    # Keep running
    try:
        await sensor_task
    except KeyboardInterrupt:
        sensor_task.cancel()
        log.info("Sensor gateway shutdown")

if __name__ == "__main__":
    import uvicorn
    # Run FastAPI and sensor loop together
    config = uvicorn.Config(
        "sensor_gateway:app", 
        host="0.0.0.0", 
        port=int(os.getenv("SENSOR_GATEWAY_PORT", "9011")),
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Start both the FastAPI server and sensor loop
    loop.create_task(main())
    loop.run_until_complete(server.serve())