#!/usr/bin/env python3
"""Verify the Krea web queue serializes jobs and honours every queue control.

Runs the real krea_web app against a stub backend HTTP server so queue
ordering, front-insertion, removal, clearing and cancellation are exercised
without loading the model or touching the GPU.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import krea_web
from fastapi.testclient import TestClient


ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
STUB_GENERATION_SECONDS = 1.5

stub_state = {
    "generation_requests": [],
    "cancel_requests": 0,
    "cancel_flag_set": threading.Event(),
}


STUB_CAPABILITIES = {
    "samplers": ["euler", "res_2s", "dpm++2m"],
    "schedulers": ["discrete", "beta", "karras"],
    "upscalers": [{"name": "Latent"}, {"name": "Lanczos"}],
    "limits": {"min_width": 64, "max_width": 4096, "min_height": 64, "max_height": 4096},
    "loras": [{"name": "krea2_raw_to_turbo_r256", "path": "krea2_raw_to_turbo_r256.safetensors"}],
    "model": {"name": "stub-model.safetensors"},
}


class StubKreaBackendHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path == "/sdcpp/v1/capabilities":
            self.respond_with_json(STUB_CAPABILITIES)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(body_length)

        if self.path == "/sdcpp/v1/cancel":
            stub_state["cancel_requests"] += 1
            stub_state["cancel_flag_set"].set()
            self.respond_with_json({"status": "cancelling"})
            return

        if self.path == "/sdapi/v1/txt2img":
            stub_state["generation_requests"].append(json.loads(raw_body))
            deadline = time.monotonic() + STUB_GENERATION_SECONDS
            while time.monotonic() < deadline:
                if stub_state["cancel_flag_set"].is_set():
                    stub_state["cancel_flag_set"].clear()
                    self.respond_with_json({"images": [], "cancelled": True})
                    return
                time.sleep(0.05)
            # Do not let a cancellation that raced the stub's completion leak
            # into the next queued request.
            stub_state["cancel_flag_set"].clear()
            self.respond_with_json({"images": [ONE_PIXEL_PNG_BASE64]})
            return

        self.send_error(404)

    def respond_with_json(self, payload: dict) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args) -> None:
        pass


def build_job_params(prompt: str) -> dict:
    return {
        "prompt": prompt, "negative_prompt": "",
        "width": 64, "height": 64, "steps": 1, "cfg": 1.0, "seed": 1,
        "lora_strength": 0.6, "sampler": "res_2s", "scheduler": "beta",
        "flow_shift": 1.15, "extra_sample_args": "", "vae_tile_size": 32,
        "pag_enabled": True, "pag_scale": 0.8, "pag_layers": "7,9",
        "pag_start": 0.1, "pag_end": 0.9,
        "hires": False, "hires_width": 128, "hires_height": 128,
        "hires_steps": 1, "hires_denoise": 0.5, "save_lowres": False,
        "use_turbo_lora": True,
    }


def find_generation_request_for_prompt(prompt: str) -> dict:
    for request_body in stub_state["generation_requests"]:
        if request_body["prompt"].startswith(prompt):
            return request_body
    raise AssertionError(f"no backend generation request was sent for prompt {prompt!r}")


def wait_until(condition, timeout_seconds: float, description: str):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {description}")


def check_server_log_tail_reader(check) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as handle:
        handle.write("first line\n")
        handle.write("second line\n")
        handle.write("  |####     | 1/6 - 1.65s/it\x1b[K\r  |########| 6/6 - 1.65s/it\x1b[K\n")
        handle.write("[INFO ] sampling completed\n")
        log_path = Path(handle.name)

    all_lines = krea_web.read_last_log_lines(log_path, 100)
    stripped_lines = [line.strip() for line in all_lines]
    check("log tail splits carriage-return progress updates into separate lines",
          "|####     | 1/6 - 1.65s/it" in stripped_lines
          and "|########| 6/6 - 1.65s/it" in stripped_lines,
          f"lines={all_lines[-4:]}")
    check("log tail strips ANSI escape sequences",
          not any("\x1b" in line for line in all_lines), f"lines={all_lines[-4:]}")
    check("log tail returns the newest line last",
          all_lines[-1] == "[INFO ] sampling completed", f"last={all_lines[-1]!r}")
    check("log tail honours the requested line count",
          len(krea_web.read_last_log_lines(log_path, 2)) == 2)
    check("log tail on a missing file returns no lines",
          krea_web.read_last_log_lines(Path("/nonexistent/krea.log"), 5) == [])

    if krea_web.SERVER_LOG_FILE.is_file():
        real_lines = krea_web.read_last_log_lines(krea_web.SERVER_LOG_FILE, 40)
        check("real server log tails without control bytes or oversized lines",
              real_lines and not any("\x1b" in line or "\r" in line for line in real_lines)
              and max(len(line) for line in real_lines) < 400,
              f"longest={max((len(l) for l in real_lines), default=0)}")
    log_path.unlink()


def main() -> int:
    stub_server = ThreadingHTTPServer(("127.0.0.1", 0), StubKreaBackendHandler)
    threading.Thread(target=stub_server.serve_forever, daemon=True).start()
    backend_url = f"http://127.0.0.1:{stub_server.server_address[1]}"

    app = krea_web.build_app("testuser", "testpass", backend_url)
    client = TestClient(app)
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"{'PASS' if condition else 'FAIL'}: {label} {detail}".rstrip())
        if not condition:
            failures.append(label)

    check("the page itself loads without credentials so the login form can render",
          client.get("/").status_code == 200)
    page_text = client.get("/").text
    check("the page exposes PAG and the requested portrait resolution",
          'id="pag_enabled"' in page_text and "1248x1824" in page_text)
    check("login rejects a wrong password",
          client.post("/api/login", json={"username": "testuser", "password": "wrong"}).status_code == 401)
    check("login rejects a wrong username",
          client.post("/api/login", json={"username": "nobody", "password": "testpass"}).status_code == 401)

    login_response = client.post("/api/login", json={"username": "testuser", "password": "testpass"})
    check("login with the correct password issues a token", login_response.status_code == 200)
    session_token = login_response.json()["token"]
    session_headers = {"Authorization": f"Bearer {session_token}"}
    check("issued token is not a guessable value", len(session_token) >= 32, f"len={len(session_token)}")
    check("login does not set a persistent cookie, so a reload must log in again",
          not login_response.cookies, f"cookies={dict(login_response.cookies)}")

    check("unauthenticated request is rejected", client.get("/api/state").status_code == 401)
    check("a bogus bearer token is rejected",
          client.get("/api/state", headers={"Authorization": "Bearer not-a-real-token"}).status_code == 401)
    check("presets endpoint lists Turbo 1024 → 2048",
          "Turbo 1024 → 2048" in client.get("/api/presets", headers=session_headers).json())
    check("prompt is required",
          client.post("/api/jobs", json={"params": build_job_params("")}, headers=session_headers).status_code == 400)

    first_job = client.post("/api/jobs", json={"params": build_job_params("first")}, headers=session_headers).json()
    second_job = client.post("/api/jobs", json={"params": build_job_params("second")}, headers=session_headers).json()
    third_job = client.post("/api/jobs", json={"params": build_job_params("third"), "front": True}, headers=session_headers).json()

    state = wait_until(lambda: client.get("/api/state", headers=session_headers).json().get("current"),
                       5, "first job to start running")
    check("only one job runs at a time", state["id"] == first_job["id"], f"running={state['id']}")

    queued_ids = [j["id"] for j in client.get("/api/state", headers=session_headers).json()["queued"]]
    check("queue-in-front puts the newest job ahead of the waiting one",
          queued_ids == [third_job["id"], second_job["id"]], f"queued={queued_ids}")

    remove_response = client.post(f"/api/jobs/{second_job['id']}/cancel", headers=session_headers)
    check("removing a waiting job succeeds", remove_response.status_code == 200)
    check("removing a running job is refused",
          client.post(f"/api/jobs/{first_job['id']}/cancel", headers=session_headers).status_code == 409)

    queued_ids = [j["id"] for j in client.get("/api/state", headers=session_headers).json()["queued"]]
    check("removed job leaves the queue", queued_ids == [third_job["id"]], f"queued={queued_ids}")

    cancel_response = client.post("/api/cancel-current", headers=session_headers).json()
    check("cancel-current reports cancelling", cancel_response.get("cancelling") is True)
    check("cancel reached the backend", stub_state["cancel_requests"] >= 1,
          f"cancel_requests={stub_state['cancel_requests']}")

    finished_first = wait_until(
        lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                      if j["id"] == first_job["id"]), None),
        10, "first job to finish after cancellation")
    check("cancelled running job is not marked completed",
          finished_first["status"] in {"cancelled", "failed"}, f"status={finished_first['status']}")

    third_finished = wait_until(
        lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                      if j["id"] == third_job["id"]), None),
        20, "front-inserted job to complete")
    check("front-inserted job completes and writes an output",
          third_finished["status"] == "completed" and bool(third_finished["outputs"]),
          f"status={third_finished['status']} outputs={third_finished['outputs']} error={third_finished['error']}")

    output_name = third_finished["outputs"][0]
    check("output filename is namespaced by job id", third_job["id"] in output_name, output_name)
    check("output is downloadable through auth",
          client.get(f"/api/output/{output_name}", headers=session_headers).status_code == 200)
    check("gallery images authenticate by token query, since img tags cannot send headers",
          client.get(f"/api/output/{output_name}?token={session_token}").status_code == 200)
    check("gallery image request without a token is rejected",
          client.get(f"/api/output/{output_name}").status_code == 401)
    check("output path traversal is blocked",
          client.get("/api/output/..%2F..%2Fetc%2Fpasswd", headers=session_headers).status_code in {404, 400})

    for extra_index in range(3):
        client.post("/api/jobs", json={"params": build_job_params(f"bulk{extra_index}")}, headers=session_headers)
    wait_until(lambda: client.get("/api/state", headers=session_headers).json()["queue_count"] >= 1,
               5, "bulk jobs to queue")
    kill_all_response = client.post("/api/kill-all", headers=session_headers).json()
    check("kill-all clears the waiting queue", kill_all_response.get("cleared", 0) >= 1,
          f"response={kill_all_response}")
    wait_until(lambda: client.get("/api/state", headers=session_headers).json()["queue_count"] == 0,
               10, "queue to drain after kill-all")
    check("queue count is zero after kill-all",
          client.get("/api/state", headers=session_headers).json()["queue_count"] == 0)

    generated_prompts = [r["prompt"].split(" <sd_cpp_extra_args>")[0]
                         for r in stub_state["generation_requests"]]
    check("removed job never reached the backend", "second" not in generated_prompts,
          f"prompts={generated_prompts}")

    backend_options = client.get("/api/backend-options", headers=session_headers).json()
    check("backend options expose the runtime sampler list",
          backend_options["samplers"] == STUB_CAPABILITIES["samplers"],
          f"samplers={backend_options.get('samplers')}")
    check("backend options expose the runtime scheduler list",
          backend_options["schedulers"] == STUB_CAPABILITIES["schedulers"],
          f"schedulers={backend_options.get('schedulers')}")
    check("backend options expose upscaler names as plain strings",
          backend_options["upscalers"] == ["Latent", "Lanczos"],
          f"upscalers={backend_options.get('upscalers')}")
    check("backend options expose the resolution limits",
          backend_options["limits"]["max_width"] == 4096)

    turbo_on_request = find_generation_request_for_prompt("third")
    native_args = json.loads(turbo_on_request["prompt"].split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("PAG settings are sent in native main-stage sample params",
          native_args["sample_params"].get("pag") == {
              "enabled": True, "scale": 0.8, "layers": [7, 9], "start": 0.1, "end": 0.9,
          }, f"pag={native_args['sample_params'].get('pag')}")
    check("turbo LoRA is sent when the checkbox is on",
          turbo_on_request.get("lora") == [{"path": krea_web.TURBO_LORA_FILENAME, "multiplier": 0.6,
                                            "is_high_noise": False}],
          f"lora={turbo_on_request.get('lora')}")

    turbo_off_params = build_job_params("turbo-off")
    turbo_off_params["use_turbo_lora"] = False
    client.post("/api/jobs", json={"params": turbo_off_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "turbo-off"
                             and j["status"] == "completed"), None),
               20, "turbo-off job to complete")
    turbo_off_request = find_generation_request_for_prompt("turbo-off")
    check("no lora key is sent when the checkbox is off",
          "lora" not in turbo_off_request, f"keys={sorted(turbo_off_request)}")

    check_server_log_tail_reader(check)
    check("server log endpoint requires authentication",
          client.get("/api/server-log").status_code == 401)
    log_response = client.get("/api/server-log?lines=3", headers=session_headers).json()
    check("server log endpoint returns at most the requested lines",
          len(log_response["lines"]) <= 3, f"count={len(log_response['lines'])}")
    check("server log endpoint clamps an oversized line request",
          len(client.get("/api/server-log?lines=99999", headers=session_headers).json()["lines"])
          <= krea_web.MAX_SERVER_LOG_LINES)

    available_loras = client.get("/api/loras", headers=session_headers).json()["loras"]
    check("lora listing exposes every safetensors file in the lora directory",
          {entry["filename"] for entry in available_loras}
          == {path.name for path in krea_web.LORA_DIR.glob("*.safetensors")
              if path.name != krea_web.TURBO_LORA_FILENAME},
          f"listed={[e['filename'] for e in available_loras]}")
    check("lora listing excludes the turbo lora, which has its own checkbox",
          all(entry["filename"] != krea_web.TURBO_LORA_FILENAME for entry in available_loras))

    if available_loras:
        extra_lora_filename = available_loras[0]["filename"]
        extra_lora_params = build_job_params("extra-loras")
        extra_lora_params["use_turbo_lora"] = True
        extra_lora_params["extra_loras"] = [{"filename": extra_lora_filename, "strength": 0.8}]
        client.post("/api/jobs", json={"params": extra_lora_params}, headers=session_headers)
        wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                                 if (j["params"] or {}).get("prompt") == "extra-loras"
                                 and j["status"] == "completed"), None),
                   20, "extra-lora job to complete")
        extra_lora_request = find_generation_request_for_prompt("extra-loras")
        sent_loras = extra_lora_request.get("lora", [])
        check("checked lora is sent with its own strength alongside the turbo lora",
              {"path": extra_lora_filename, "multiplier": 0.8, "is_high_noise": False} in sent_loras
              and any(entry["path"] == krea_web.TURBO_LORA_FILENAME for entry in sent_loras),
              f"lora={sent_loras}")

        unchecked_params = build_job_params("no-extra-loras")
        unchecked_params["extra_loras"] = []
        client.post("/api/jobs", json={"params": unchecked_params}, headers=session_headers)
        wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                                 if (j["params"] or {}).get("prompt") == "no-extra-loras"
                                 and j["status"] == "completed"), None),
                   20, "no-extra-lora job to complete")
        unchecked_request = find_generation_request_for_prompt("no-extra-loras")
        check("unchecked loras are not sent to the backend",
              [entry["path"] for entry in unchecked_request.get("lora", [])]
              == [krea_web.TURBO_LORA_FILENAME],
              f"lora={unchecked_request.get('lora')}")

    check("logout invalidates the token immediately",
          client.post("/api/logout", headers=session_headers).status_code == 200
          and client.get("/api/state", headers=session_headers).status_code == 401)

    for cleanup_name in list(krea_web.OUTPUT_DIR.glob("krea-web*")):
        if third_job["id"] in cleanup_name.name:
            cleanup_name.unlink()

    stub_server.shutdown()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all queue control checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
