import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
#from opentelemetry.exporter.gcp_trace import CloudTraceSpanExporter
def setup_tracer():
    # Skip GCP tracer in local development
    if os.getenv("ENVIRONMENT") == "local":
        return trace.get_tracer("novabuy-support-local")
    
    exporter = CloudTraceSpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("novabuy-support")

tracer = setup_tracer()

 