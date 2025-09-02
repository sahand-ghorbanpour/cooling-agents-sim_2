
import os, json, redis
def _r():
    return redis.Redis.from_url(os.getenv("REDIS_URL","redis://redis:6379/0"), decode_responses=True)
def set_config(k,v): _r().hset("config", k, json.dumps(v))
def get_config(k, default=None):
    raw=_r().hget("config", k); 
    if raw is None: return default
    try: return json.loads(raw)
    except Exception: return default
def set_latest_state(s:dict): _r().set("latest_state", json.dumps(s))
def get_latest_state(default=None):
    raw=_r().get("latest_state"); 
    if raw is None: return default
    try: return json.loads(raw)
    except Exception: return default
