
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource, Attributes
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
_inited=False
def init_tracer(service_name="service"):
    global _inited
    if _inited: return trace.get_tracer(service_name)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT","http://tempo:4317")
    provider = TracerProvider(resource=Resource(attributes={SERVICE_NAME: service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True)))
    trace.set_tracer_provider(provider); _inited=True
    return trace.get_tracer(service_name)
