import asyncio, json, os, time, logging
from nats.aio.client import Client as NATS
from nats.js.api import StreamConfig, RetentionPolicy
from typing import Optional, Dict, Any

# Global connection pool
_nats_connection: Optional[NATS] = None
_connection_lock = asyncio.Lock()
_last_health_check = 0
log = logging.getLogger(__name__)

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

async def get_nats_connection() -> NATS:
    """Get shared NATS connection with health checking"""
    global _nats_connection, _last_health_check
    
    async with _connection_lock:
        current_time = time.time()
        
        # Check if we need to create a new connection or health check
        if (_nats_connection is None or 
            not _nats_connection.is_connected or 
            current_time - _last_health_check > 30):  # Health check every 30s
            
            if _nats_connection and _nats_connection.is_connected:
                try:
                    await _nats_connection.close()
                except Exception:
                    pass
            
            # Create new connection
            _nats_connection = NATS()
            servers = os.getenv("NATS_URL", "nats://nats:4222")
            
            try:
                await _nats_connection.connect(
                    servers=servers,
                    connect_timeout=5.0,
                    max_reconnect_attempts=3,
                    reconnect_time_wait=2.0
                )
                _last_health_check = current_time
                log.debug("NATS connection established")
            except Exception as e:
                log.error(f"Failed to connect to NATS: {e}")
                raise
        
        return _nats_connection

async def get_js():
    """Get JetStream context with stream initialization"""
    nc = await get_nats_connection()
    js = nc.jetstream()
    
    # Ensure streams exist (idempotent)
    for name, subjects in {
        "DC": ["dc.*"],
        "TRIG": ["dc.trigger.*"],
        "SIM": ["simulation.*"],
        "TELE": ["dc.telemetry.*"],
        "CONTROL": ["dc.control.*"]
    }.items():
        try:
            await js.add_stream(StreamConfig(
                name=name, 
                subjects=subjects, 
                retention=RetentionPolicy.WORK_QUEUE,
                max_age=3600  # 1 hour retention
            ))
        except Exception as e:
            # Stream might already exist
            log.debug(f"Stream {name} setup: {e}")
    
    return nc, js

async def publish(subject: str, payload: Dict[Any, Any], agent: str = "sys", direction: str = "out"):
    """Publish message with retry logic and connection reuse"""
    from common.metrics import nats_messages_total, nats_msg_bytes
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            nc, js = await get_js()
            data = json.dumps(payload).encode()
            
            # Use JetStream publish with timeout
            await asyncio.wait_for(js.publish(subject, data), timeout=10.0)
            
            # Update metrics
            nats_messages_total.labels(agent, subject, direction).inc()
            nats_msg_bytes.labels(agent, subject, direction).inc(len(data))
            
            log.debug(f"Published to {subject}: {len(data)} bytes")
            return
            
        except asyncio.TimeoutError:
            log.warning(f"NATS publish timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            continue
            
        except Exception as e:
            log.error(f"NATS publish failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # Force connection refresh on error
                global _nats_connection
                async with _connection_lock:
                    if _nats_connection:
                        try:
                            await _nats_connection.close()
                        except Exception:
                            pass
                        _nats_connection = None
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            continue
    
    raise Exception(f"Failed to publish to {subject} after {max_retries} attempts")

async def close_connection():
    """Gracefully close NATS connection"""
    global _nats_connection
    async with _connection_lock:
        if _nats_connection and _nats_connection.is_connected:
            try:
                await _nats_connection.close()
                log.info("NATS connection closed")
            except Exception as e:
                log.warning(f"Error closing NATS connection: {e}")
            finally:
                _nats_connection = None