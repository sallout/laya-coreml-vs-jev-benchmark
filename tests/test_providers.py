import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from laya_jev_bench.providers import JevProvider


class Handler(BaseHTTPRequestHandler):
    request_payload = None

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        Handler.request_payload = json.loads(self.rfile.read(length))
        payload = {
            "model": "jev-test",
            "answers": {
                "intent": {
                    "choice": "billing",
                    "probabilities": {"billing": 0.8, "other": 0.2},
                    "confidence": 0.7,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class ProviderTests(unittest.TestCase):
    def test_jev_request_contract(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with patch.dict(os.environ, {"TEST_TYPESAFE_KEY": "secret"}):
                provider = JevProvider(
                    "jev-test",
                    "TEST_TYPESAFE_KEY",
                    f"http://127.0.0.1:{server.server_port}/v1/systemone",
                    "typesafe",
                    2,
                    0,
                )
                question = {
                    "type": "choice",
                    "instructions": "Choose one.",
                    "criteria": {"billing": None, "other": None},
                }
                decision = provider.decide("charged twice", question)
            self.assertEqual(decision.choice, "billing")
            self.assertEqual(
                Handler.request_payload,
                {
                    "model": "jev-test",
                    "state": "charged twice",
                    "questions": {"intent": question},
                },
            )
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
