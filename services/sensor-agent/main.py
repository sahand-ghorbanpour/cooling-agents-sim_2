# services/sensor-agent/main.py - Fixed JSON serialization and optimized flow
import asyncio, json, os, time, logging
import numpy as np
from nats.aio.client import Client as NATS
from prometheus_client import start_http_server, Counter, Histogram
from common.logging import setup_json_logging
from common.config import get_config, set_latest_state
from typing import Dict
# Import your physics simulation
from env.frontier_env import SmallFrontierModel

log = setup_json_logging()
start_http_server(int(os.getenv("METRICS_PORT", "9012")))

# Metrics
simulation_steps = Counter('simulation_steps_total', 'Total simulation steps')
simulation_duration = Histogram('simulation_step_duration_seconds', 'Time per simulation step')
simulation_errors = Counter('simulation_errors_total', 'Simulation errors', ['error_type'])

def numpy_to_json_safe(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-safe format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_json_safe(item) for item in obj]
    else:
        return obj

class SensorAgentService:
    def __init__(self):
        self.env = None
        self.nc = None
        self.current_actions = None
        self.step_count = 0
        self.last_state = {}
        
    async def initialize(self):
        """Initialize simulation environment and NATS connection"""
        try:
            # Initialize FMU environment
            self.env = SmallFrontierModel(
                start_time=0,
                stop_time=86400,  # 24 hours
                step_size=15.0,   # 15 seconds
                use_reward_shaping='reward_shaping_v1'
            )
            initial_state, _ = self.env.reset()
            log.info("SmallFrontierModel initialized successfully")
            
        except Exception as e:
            log.error(f"Failed to initialize SmallFrontierModel: {e}")
            simulation_errors.labels(error_type='initialization').inc()
            raise
        
        # Connect to NATS
        self.nc = NATS()
        await self.nc.connect(servers=os.getenv("NATS_URL","nats://nats:4222"))
        
        # Subscribe to control actions from control system  
        await self.nc.subscribe("dc.control.actions", cb=self.handle_control_actions)
        
        log.info("Sensor agent service initialized and subscribed to NATS")
    
    async def handle_control_actions(self, msg):
        """Handle control actions from control runtime"""
        try:
            payload = json.loads(msg.data.decode())
            data = payload.get("data", {})
            
            # Store actions for next simulation step
            self.current_actions = data
            log.debug("Received control actions for next simulation step")
            
        except Exception as e:
            log.error(f"Error handling control actions: {e}")
    
    async def step_simulation(self):
        """Execute one simulation step"""
        if not self.env:
            log.error("Simulation environment not initialized")
            return
        
        with simulation_duration.time():
            try:
                # Use received actions or defaults
                if self.current_actions:
                    actions = self.current_actions
                    self.current_actions = None
                    
                    # Convert lists to numpy arrays for simulation environment
                    for key, action in actions.items():
                        if isinstance(action, (list, tuple)) and key.startswith('cdu-cabinet-'):
                            actions[key] = np.array(action, dtype=np.float32)
                    
                else:
                    # Default actions - maintain current state
                    actions = {
                        'cdu-cabinet-1': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-2': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-3': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-4': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cdu-cabinet-5': np.array([0.0, 0.0, 1/3, 1/3, 1/3], dtype=np.float32),
                        'cooling-tower-1': 4
                    }
                
                # Step simulation
                obs, rewards, terminateds, truncateds, info = self.env.step(actions)
                
                # Extract temperature data
                temps = {}
                for i in range(1, 6):
                    cabinet_key = f"cdu-cabinet-{i}"
                    if cabinet_key in obs:
                        cabinet_obs = obs[cabinet_key]
                        if len(cabinet_obs) >= 3:
                            # Convert normalized observations to Celsius
                            temp_norm = cabinet_obs[:3]
                            temp_k = ((temp_norm + 1) / 2) * (313.15 - 293.15) + 293.15
                            avg_temp_c = float(np.mean(temp_k) - 273.15)
                            temps[f"cabinet_{i}"] = avg_temp_c
                
                # Extract cooling tower data
                if "cooling-tower-1" in obs:
                    ct_obs = obs["cooling-tower-1"]
                    if len(ct_obs) >= 3:
                        water_temp_norm = ct_obs[2]
                        water_temp_k = ((water_temp_norm + 1) / 2) * (313.15 - 293.15) + 293.15
                        temps["cooling_tower"] = float(water_temp_k - 273.15)
                
                # Calculate energy consumption
                total_energy = 0.0
                if "cooling-tower-1" in obs:
                    ct_obs = obs["cooling-tower-1"]
                    if len(ct_obs) >= 2:
                        fan_power = np.sum(ct_obs[:2])
                        total_energy = float((fan_power + 2) / 4 * 150)
                
                # Prepare simulation state with JSON-safe conversion
                simulation_state = {
                    "temps": temps,
                    "meta": {
                        "energy_kw": total_energy,
                        "step_count": self.step_count,
                        "rewards": {k: float(v) for k, v in rewards.items()},
                        "simulation_time": self.env.current_time if hasattr(self.env, 'current_time') else self.step_count * 15.0,
                        "actions_applied": numpy_to_json_safe(actions)  # Convert numpy arrays
                    },
                    "timestamp": time.time(),
                    "source": "sensor_agent_service"
                }
                
                # Publish optimized - directly to final destination
                await self.publish_optimized_state(simulation_state)
                
                # Update internal state
                self.last_state = simulation_state
                self.step_count += 1
                simulation_steps.inc()
                
                log.debug(f"Simulation step {self.step_count} completed")
                
            except Exception as e:
                log.error(f"Simulation step failed: {e}")
                simulation_errors.labels(error_type='simulation_step').inc()
                
                # Publish fallback state
                fallback_state = {
                    "temps": {f"cabinet_{i}": 25.0 for i in range(1,6)},
                    "meta": {"error": str(e), "step_count": self.step_count},
                    "timestamp": time.time(),
                    "source": "sensor_agent_service_fallback"
                }
                await self.publish_optimized_state(fallback_state)
    
    async def publish_optimized_state(self, state: Dict):
        """Optimized publishing - direct to destinations, no intermediate hops"""
        try:
            # Convert to JSON-safe format
            json_safe_state = numpy_to_json_safe(state)
            state_json = json.dumps(json_safe_state).encode()
            
            # Publish directly to final destinations (eliminates bridge hop)
            await self.nc.publish("dc.telemetry.state", state_json)
            
            # Store directly in Redis (eliminates graph processing hop)
            clean_data = {
                "temps": json_safe_state.get("temps", {}),
                "meta": json_safe_state.get("meta", {}),
                "ts": json_safe_state.get("timestamp", time.time())
            }
            set_latest_state(clean_data)
            
            log.debug("Published optimized simulation state")
            
        except Exception as e:
            log.error(f"Failed to publish simulation state: {e}")
            simulation_errors.labels(error_type='publish_failed').inc()
    
    async def run_autonomous_mode(self):
        """Run simulation autonomously when not triggered"""
        log.info("Starting autonomous simulation mode")
        
        while True:
            try:
                # Check if we should run autonomous simulation
                autonomous_interval = float(get_config("autonomous_simulation_interval_ms", 5000)) / 1000.0
                
                # Step simulation autonomously
                await self.step_simulation()
                
                await asyncio.sleep(autonomous_interval)
                
            except Exception as e:
                log.error(f"Autonomous simulation error: {e}")
                await asyncio.sleep(5.0)

async def main():
    """Main service entry point"""
    service = SensorAgentService()
    await service.initialize()
    
    # Start autonomous simulation as background task
    autonomous_task = asyncio.create_task(service.run_autonomous_mode())
    
    log.info("Sensor agent service running...")
    
    try:
        await autonomous_task
    except KeyboardInterrupt:
        autonomous_task.cancel()
        if service.nc and service.nc.is_connected:
            await service.nc.close()
        log.info("Sensor agent service shutdown")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Sensor agent stopped by user")
    except Exception as e:
        log.error(f"Sensor agent startup failed: {e}")
        exit(1)