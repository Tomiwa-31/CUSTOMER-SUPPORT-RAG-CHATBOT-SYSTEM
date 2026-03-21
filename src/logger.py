# src/logger.py

import json
import uuid
import time
from datetime import datetime

#Invocation ID — tracks one single request
#Every single message gets a fresh invocation ID regardless of who sent it.
def generate_invocation_id() -> str:#generate a unique id anytime user sends a requst to the chat endpoint
    return str(uuid.uuid4())[:8]


#setting our we pass in/record the log data structure for each log entry
def log(event: str, invocation_id: str = None, **kwargs):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "invocation_id": invocation_id or "system",
        "event": event,
        **kwargs
    }
    print(json.dumps(entry))#outputs it in a json format


class RequestLogger:
    def __init__(self):
        self.invocation_id = generate_invocation_id()
        self.start_time = time.time()

    def latency_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)

    def info(self, event: str, **kwargs):
        log(event, invocation_id=self.invocation_id, level="INFO", **kwargs)

    def warning(self, event: str, **kwargs):
        log(event, invocation_id=self.invocation_id, level="WARNING", **kwargs)

    def error(self, event: str, **kwargs):
        log(event, invocation_id=self.invocation_id, level="ERROR", **kwargs)

    #one of three levels of logging,
    #info for normal events(like request received, request completed)
    #warning for non-critical issues (like slow response times)
    #error for critical issues (like exceptions)


    #session_id: "user123" (stays same throughout)
    #├── invocation_id: "a3f9bc12"  → "what is the return window?"
    #├── invocation_id: "b7d2ef45"  → "what about exchanges?"
    #└── invocation_id: "c9e3fg67"  → "how long does refund take?"

    #Session ID — tracks a user across multiple requests