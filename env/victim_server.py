"""Threaded HTTP service for the victim host (h16).

Replaces `python -m http.server`, which is single-threaded: under thirty probe
connections a second its accept queue backs up and benign probes fail, which
made the R0 baseline unusable. This server accepts concurrently and keeps a
generous listen backlog, so benign service stays near 1.0 and any degradation
the experiment measures comes from the attack, not the server.

A SYN flood still degrades service: it exhausts the kernel half-open (SYN)
queue, which benign completed connections never occupy, so the two effects stay
separate. Run on the victim as: python3 env/victim_server.py --port 80
"""

from __future__ import annotations

import argparse
import http.server
import socketserver


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    request_queue_size = 128        # generous accept backlog for benign probes
    allow_reuse_address = True


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D401 - silence per-request logging
        pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=80)
    args = p.parse_args()
    with _Server(("0.0.0.0", args.port), _Handler) as srv:
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
