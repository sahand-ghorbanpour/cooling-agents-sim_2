from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import numpy as np
import os, time, logging
from prometheus_client import start_http_server, Counter, Histogram, Gauge
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from common.otel import init_tracer
from common.logging import setup_json_logging

log = setup_json_logging()
app = FastAPI(title="MCP Math - Advanced Thermal Optimization")
FastAPIInstrumentor.instrument_app(app)
init_tracer("mcp-math")
start_http_server(int(os.getenv("METRICS_PORT","9004")))

# Prometheus metrics
optimization_requests = Counter('thermal_optimization_requests_total', 'Total optimization requests', ['strategy'])
optimization_duration = Histogram('thermal_optimization_duration_seconds', 'Time spent on optimization')
optimization_confidence = Gauge('thermal_optimization_confidence', 'Confidence score of last optimization')
temperature_deviation = Gauge('thermal_temperature_deviation_celsius', 'Temperature deviation from target')
energy_efficiency_score = Gauge('thermal_energy_efficiency_score', 'Energy efficiency score (0-1)')

# Constants from your original project
TARGET_TEMP_K = float(os.getenv("TARGET_TEMP_K", "303.15"))  # 30°C
TARGET_TEMP_C = TARGET_TEMP_K - 273.15
TEMP_MIN_K = 293.15  # 20°C
TEMP_MAX_K = 313.15  # 40°C
TEMP_TOLERANCE = float(os.getenv("TEMP_TOLERANCE", "0.5"))

class ThermalOptimizationRequest(BaseModel):
    temps: Dict[str, float] = Field(..., description="Current temperature readings")
    target_temp_c: float = Field(default=TARGET_TEMP_C, description="Target temperature in Celsius")
    energy_weight: float = Field(default=0.3, description="Weight for energy efficiency (0-1)")
    stability_weight: float = Field(default=0.7, description="Weight for temperature stability (0-1)")
    current_actions: Optional[Dict[str, Any]] = Field(default=None, description="Current control actions")
    history: Optional[List[Dict]] = Field(default=None, description="Historical data for trend analysis")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="System constraints")

class ThermalOptimizationResult(BaseModel):
    strategy: str = Field(..., description="Chosen optimization strategy")
    cabinet_actions: Dict[str, List[float]] = Field(..., description="Cabinet control actions")
    cooling_tower_action: int = Field(..., description="Cooling tower action [0-8]")
    confidence: float = Field(..., description="Confidence in the solution [0-1]")
    energy_score: float = Field(..., description="Energy efficiency score [0-1]")
    temperature_score: float = Field(..., description="Temperature control score [0-1]")
    optimization_time_ms: float = Field(..., description="Optimization time in milliseconds")
    reasoning: str = Field(..., description="Explanation of the optimization decision")

# Cooling tower action mapping from your original project
COOLING_TOWER_ACTION_DECODING = {
    0: -0.20, 1: -0.15, 2: -0.10, 3: -0.05, 4: 0,
    5: 0.05, 6: 0.10, 7: 0.15, 8: 0.20
}

@app.get("/healthz")
def health_check():
    return {"status": "healthy", "service": "mcp-math-thermal", "timestamp": time.time()}

@app.post("/thermal_optimize", response_model=ThermalOptimizationResult)
def thermal_optimize(request: ThermalOptimizationRequest):
    """Advanced thermal optimization using your original algorithms"""
    start_time = time.time()
    
    try:
        with optimization_duration.time():
            # Extract current temperatures
            cabinet_temps = {}
            cooling_tower_temp = request.target_temp_c
            
            for key, temp in request.temps.items():
                if key.startswith("cabinet_"):
                    cabinet_temps[key] = temp
                elif key == "cooling_tower":
                    cooling_tower_temp = temp
            
            if not cabinet_temps:
                raise HTTPException(status_code=400, detail="No cabinet temperatures provided")
            
            # Calculate system metrics
            global_temp = sum(cabinet_temps.values()) / len(cabinet_temps)
            temp_deviation = global_temp - request.target_temp_c
            temperature_deviation.set(abs(temp_deviation))
            
            # Determine control strategy based on deviation
            strategy, reasoning = _determine_strategy(temp_deviation, cabinet_temps, request.target_temp_c)
            
            # Generate cabinet actions using your original reward shaping logic
            cabinet_actions = {}
            for i in range(1, 6):
                cabinet_key = f"cabinet_{i}"
                cabinet_temp = cabinet_temps.get(cabinet_key, global_temp)
                action = _generate_cabinet_action(
                    cabinet_temp, 
                    request.target_temp_c, 
                    temp_deviation,
                    strategy,
                    request.energy_weight
                )
                cabinet_actions[f"cdu-cabinet-{i}"] = action
            
            # Generate cooling tower action
            ct_action = _generate_cooling_tower_action(
                global_temp, 
                request.target_temp_c, 
                strategy,
                cooling_tower_temp
            )
            
            # Calculate performance scores
            energy_score = _calculate_energy_score(ct_action, cabinet_actions)
            temperature_score = _calculate_temperature_score(temp_deviation)
            
            # Confidence based on strategy and deviation
            confidence = _calculate_confidence(strategy, temp_deviation)
            
            # Update metrics
            optimization_requests.labels(strategy=strategy).inc()
            optimization_confidence.set(confidence)
            energy_efficiency_score.set(energy_score)
            
            optimization_time_ms = (time.time() - start_time) * 1000
            
            log.info(f"Thermal optimization: strategy={strategy}, deviation={temp_deviation:.2f}C, confidence={confidence:.3f}")
            
            return ThermalOptimizationResult(
                strategy=strategy,
                cabinet_actions=cabinet_actions,
                cooling_tower_action=ct_action,
                confidence=confidence,
                energy_score=energy_score,
                temperature_score=temperature_score,
                optimization_time_ms=optimization_time_ms,
                reasoning=reasoning
            )
            
    except Exception as e:
        log.error(f"Thermal optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/legacy_optimize")
def legacy_optimize(request: Dict[str, Any]):
    """Legacy endpoint compatible with your old system"""
    temps = request.get("temps", {})
    
    # Convert to new format
    thermal_request = ThermalOptimizationRequest(
        temps=temps,
        target_temp_c=request.get("target", TARGET_TEMP_C),
        energy_weight=request.get("energy_weight", 0.3),
        stability_weight=request.get("stability_weight", 0.7)
    )
    
    result = thermal_optimize(thermal_request)
    
    # Convert back to legacy format
    legacy_actions = {}
    for cabinet_key, action_list in result.cabinet_actions.items():
        # Convert to fan percentage (simplified)
        temp_adjustment = action_list[0]  # Secondary temperature adjustment
        fan_pct = max(10.0, min(100.0, 50.0 - temp_adjustment * 50.0))
        legacy_key = cabinet_key.replace("cdu-cabinet-", "cabinet_") + "_fan_pct"
        legacy_actions[legacy_key] = fan_pct
    
    # Cooling tower speed
    ct_normalized = COOLING_TOWER_ACTION_DECODING[result.cooling_tower_action]
    ct_speed_pct = max(10.0, min(100.0, 50.0 + ct_normalized * 50.0))
    legacy_actions["cooling_tower_speed_pct"] = ct_speed_pct
    
    return {
        "targets": legacy_actions,
        "strategy": result.strategy,
        "confidence": result.confidence,
        "energy_score": result.energy_score
    }

def _determine_strategy(temp_deviation: float, cabinet_temps: Dict[str, float], target_temp: float) -> tuple[str, str]:
    """Determine control strategy based on temperature deviation"""
    abs_dev = abs(temp_deviation)
    
    if abs_dev <= TEMP_TOLERANCE:
        strategy = "energy_optimize"
        reasoning = f"Temperature within tolerance (±{TEMP_TOLERANCE}°C), optimizing for energy efficiency"
    elif abs_dev > 3.0:
        strategy = "emergency_cooling" if temp_deviation > 0 else "emergency_heating"
        reasoning = f"Critical temperature deviation ({temp_deviation:.2f}°C), emergency response required"
    elif abs_dev > 1.5:
        strategy = "aggressive_correction"
        reasoning = f"High temperature deviation ({temp_deviation:.2f}°C), aggressive correction needed"
    elif abs_dev > TEMP_TOLERANCE:
        strategy = "proportional_control"
        reasoning = f"Moderate temperature deviation ({temp_deviation:.2f}°C), proportional control applied"
    else:
        strategy = "maintain_stability"
        reasoning = "Temperature stable, maintaining current control strategy"
    
    return strategy, reasoning

def _generate_cabinet_action(cabinet_temp: float, target_temp: float, global_deviation: float, 
                           strategy: str, energy_weight: float) -> List[float]:
    """Generate cabinet control action [sec_temp, pressure, valve1, valve2, valve3]"""
    
    local_deviation = cabinet_temp - target_temp
    
    if strategy == "emergency_cooling":
        sec_temp_adjustment = max(-0.9, -abs(local_deviation) * 0.4)
        pressure_diff = 0.3
    elif strategy == "emergency_heating":
        sec_temp_adjustment = min(0.9, abs(local_deviation) * 0.4)
        pressure_diff = -0.2
    elif strategy == "aggressive_correction":
        sec_temp_adjustment = np.clip(-local_deviation * 0.3, -0.7, 0.7)
        pressure_diff = np.clip(local_deviation * 0.1, -0.3, 0.3)
    elif strategy == "proportional_control":
        sec_temp_adjustment = np.clip(-local_deviation * 0.2, -0.5, 0.5)
        pressure_diff = np.clip(local_deviation * 0.05, -0.2, 0.2)
    elif strategy == "energy_optimize":
        # Fine tuning with energy consideration
        sec_temp_adjustment = np.tanh(-local_deviation * 0.1) * (1.0 - energy_weight)
        pressure_diff = 0.1
    else:  # maintain_stability
        sec_temp_adjustment = np.clip(-local_deviation * 0.1, -0.2, 0.2)
        pressure_diff = 0.0
    
    # Valve positions - implement power-based distribution logic from your original project
    if abs(local_deviation) < 0.5:
        # Balanced distribution for stable temperatures
        valve_positions = [1/3, 1/3, 1/3]
    else:
        # Weighted distribution based on cooling need
        cooling_need = max(0.1, min(0.9, (local_deviation + 2.0) / 4.0))
        valve_positions = [
            cooling_need * 0.4 + 0.1,      # Primary cooling
            cooling_need * 0.4 + 0.2,      # Secondary cooling  
            1.0 - (cooling_need * 0.8 + 0.3)  # Bypass
        ]
        # Normalize
        valve_sum = sum(valve_positions)
        valve_positions = [v / valve_sum for v in valve_positions]
    
    return [
        float(sec_temp_adjustment),
        float(pressure_diff),
        float(valve_positions[0]),
        float(valve_positions[1]), 
        float(valve_positions[2])
    ]

def _generate_cooling_tower_action(global_temp: float, target_temp: float, 
                                 strategy: str, cooling_tower_temp: float) -> int:
    """Generate cooling tower discrete action [0-8]"""
    
    temp_deviation = global_temp - target_temp
    
    if strategy == "emergency_cooling":
        return min(8, 6 + int(abs(temp_deviation) - 2.0))
    elif strategy == "emergency_heating":
        return max(0, 2 - int(abs(temp_deviation) - 2.0))
    elif strategy == "aggressive_correction":
        if temp_deviation > 0:
            return min(8, 5 + int(temp_deviation))
        else:
            return max(0, 3 + int(temp_deviation))
    elif strategy == "proportional_control":
        action = 4 + int(temp_deviation * 2)  # Proportional to deviation
        return max(0, min(8, action))
    elif strategy == "energy_optimize":
        # Conservative approach for energy efficiency
        if abs(temp_deviation) < 0.2:
            return 3  # Reduce energy
        elif temp_deviation > 0.5:
            return 5
        elif temp_deviation < -0.5:
            return 3
        else:
            return 4
    else:  # maintain_stability
        return 4

def _calculate_energy_score(ct_action: int, cabinet_actions: Dict[str, List[float]]) -> float:
    """Calculate energy efficiency score [0-1]"""
    
    # Cooling tower energy (higher action = more energy)
    ct_energy_factor = 1.0 - (ct_action / 8.0) * 0.6
    
    # Cabinet energy (higher pressure and valve flow = more energy)
    cabinet_energy_factors = []
    for action in cabinet_actions.values():
        pressure_factor = 1.0 - abs(action[1]) * 0.3
        valve_factor = 1.0 - (sum(action[2:5]) - 1.0) * 0.2  # Deviation from balanced flow
        cabinet_energy_factors.append((pressure_factor + valve_factor) / 2.0)
    
    avg_cabinet_energy = sum(cabinet_energy_factors) / len(cabinet_energy_factors)
    
    return (ct_energy_factor + avg_cabinet_energy) / 2.0

def _calculate_temperature_score(temp_deviation: float) -> float:
    """Calculate temperature control score [0-1]"""
    if abs(temp_deviation) <= TEMP_TOLERANCE:
        return 1.0
    else:
        return max(0.0, 1.0 - (abs(temp_deviation) - TEMP_TOLERANCE) / 5.0)

def _calculate_confidence(strategy: str, temp_deviation: float) -> float:
    """Calculate confidence in optimization result"""
    base_confidence = {
        "energy_optimize": 0.95,
        "maintain_stability": 0.90,
        "proportional_control": 0.85,
        "aggressive_correction": 0.80,
        "emergency_cooling": 0.75,
        "emergency_heating": 0.75
    }
    
    confidence = base_confidence.get(strategy, 0.70)
    
    # Reduce confidence for extreme deviations
    if abs(temp_deviation) > 2.0:
        confidence *= 0.9
    elif abs(temp_deviation) > 4.0:
        confidence *= 0.8
        
    return confidence

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000, log_level="info")