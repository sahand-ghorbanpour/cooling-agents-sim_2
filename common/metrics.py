
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import os, threading
_started=False
_lock=threading.Lock()
def init_metrics(default_port:int=9001):
    global _started
    if _started: return
    with _lock:
        if _started: return
        start_http_server(int(os.getenv("METRICS_PORT", default_port))); _started=True
graph_runs_total = Counter("graph_runs_total", "Total number of LangGraph runs", ["graph"])
node_runs_total = Counter("node_runs_total", "Total number of node executions", ["graph", "node"])
node_duration_seconds = Histogram("node_duration_seconds", "Node execution duration seconds", ["graph", "node"])

nats_messages_total = Counter("nats_messages_total", "NATS messages sent/received", ["agent","subject","direction"])
nats_msg_bytes = Counter("nats_msg_bytes_total", "NATS message bytes", ["agent","subject","direction"])
llm_request_duration_seconds = Histogram("llm_request_duration_seconds", "LLM request duration seconds", ["model"])

cooling_global_temp_c = Gauge("cooling_system_global_temp_celsius", "Global average temperature (C)")
cabinet_temp_c = Gauge("cabinet_temperature_celsius", "Cabinet temperature (C)", ["cabinet"])
energy_kw = Gauge("cooling_energy_kw", "Cooling energy usage (kW)")
efficiency_score = Gauge("cooling_efficiency_score", "Efficiency score (0-1)")

control_action = Gauge("control_action_percent", "Control action percent", ["target"])
