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
    "generation_paths": [],
    "cancel_requests": 0,
    "cancel_flag_set": threading.Event(),
}


STUB_CAPABILITIES = {
    "samplers": ["euler", "er_sde", "res_2s"],
    "schedulers": ["discrete", "beta", "karras"],
    "upscalers": [{"name": "Latent"}, {"name": "Lanczos"}],
    "limits": {"min_width": 64, "max_width": 4096, "min_height": 64, "max_height": 4096},
    "loras": [{"name": "stub-lora", "path": "stub-lora.safetensors"}],
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

        if self.path in ("/sdapi/v1/txt2img", "/sdapi/v1/img2img"):
            stub_state["generation_requests"].append(json.loads(raw_body))
            stub_state["generation_paths"].append(self.path)
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
        "lora_strength": 0.6, "sampler": "er_sde", "scheduler": "discrete",
        "flow_shift": 1.15, "extra_sample_args": "", "vae_tile_size": 32,
        "beta_schedule_alpha": 0.5, "beta_schedule_beta": 0.7,
        "pag_enabled": True, "pag_scale": 0.8, "pag_layers": "7,9",
        "pag_start": 0.1, "pag_end": 0.9,
        "hires": False, "hires_width": 128, "hires_height": 128,
        "hires_steps": 1, "hires_denoise": 0.5, "save_lowres": False,
    }


def find_generation_request_for_prompt(prompt: str) -> dict:
    for request_body in stub_state["generation_requests"]:
        if request_body["prompt"].startswith(prompt):
            return request_body
    raise AssertionError(f"no backend generation request was sent for prompt {prompt!r}")


def find_all_generation_requests_for_prompt(prompt: str) -> list[dict]:
    return [request_body for request_body in stub_state["generation_requests"]
            if request_body["prompt"].startswith(prompt)]


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

    front_inserted_request = find_generation_request_for_prompt("third")
    native_args = json.loads(front_inserted_request["prompt"].split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("PAG settings are sent in native main-stage sample params",
          native_args["sample_params"].get("pag") == {
              "enabled": True, "scale": 0.8, "layers": [7, 9], "start": 0.1, "end": 0.9,
          }, f"pag={native_args['sample_params'].get('pag')}")
    check("no lora key is sent when the user selected none",
          "lora" not in front_inserted_request, f"keys={sorted(front_inserted_request)}")
    check("beta schedule args are omitted for a non-beta scheduler",
          "alpha=" not in native_args["sample_params"]["extra_sample_args"],
          f"extra_sample_args={native_args['sample_params']['extra_sample_args']!r}")

    beta_schedule_params = build_job_params("beta-schedule")
    beta_schedule_params["scheduler"] = "beta"
    client.post("/api/jobs", json={"params": beta_schedule_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "beta-schedule"
                             and j["status"] == "completed"), None),
               20, "beta-schedule job to complete")
    beta_schedule_request = find_generation_request_for_prompt("beta-schedule")
    beta_native_args = json.loads(beta_schedule_request["prompt"].split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("beta schedule alpha and beta are sent as extra sample args for the beta scheduler",
          beta_native_args["sample_params"]["extra_sample_args"] == "alpha=0.5,beta=0.7",
          f"extra_sample_args={beta_native_args['sample_params']['extra_sample_args']!r}")

    uploaded_reference = client.post("/api/source-image",
                                     content=base64.b64decode(ONE_PIXEL_PNG_BASE64),
                                     headers={**session_headers,
                                              "Content-Type": "image/png",
                                              "X-Filename": "krea2-edit-reference.png"}).json()
    uploaded_second_reference = client.post("/api/source-image",
                                            content=base64.b64decode(ONE_PIXEL_PNG_BASE64),
                                            headers={**session_headers,
                                                     "Content-Type": "image/png",
                                                     "X-Filename": "krea2-edit-subject.png"}).json()
    krea2_edit_params = build_job_params("krea2-edit")
    krea2_edit_params["krea2_edit_enabled"] = True
    krea2_edit_params["krea2_edit_references"] = [
        {"filename": uploaded_reference["name"], "ref_boost": 1},
        {"filename": uploaded_second_reference["name"], "ref_boost": 4},
    ]
    krea2_edit_params["grounding_px"] = 768
    client.post("/api/jobs", json={"params": krea2_edit_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "krea2-edit"
                             and j["status"] == "completed"), None),
               20, "krea2-edit job to complete")
    krea2_edit_request = find_generation_request_for_prompt("krea2-edit")
    check("krea2 edit sends every reference as an extra image",
          len(krea2_edit_request.get("extra_images", [])) == 2,
          f"extra_images count={len(krea2_edit_request.get('extra_images', []))}")
    krea2_edit_native_args = json.loads(krea2_edit_request["prompt"].split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("krea2 edit selects the preset and one ref_boost per reference, in order",
          krea2_edit_native_args.get("ref_image_args")
          == "preset=krea2_edit,vlm_size=768,ref_boost=1,ref_boost=4",
          f"ref_image_args={krea2_edit_native_args.get('ref_image_args')!r}")
    check("krea2 edit uses txt2img, since the target starts as pure noise",
          stub_state["generation_paths"][-1] == "/sdapi/v1/txt2img",
          f"path={stub_state['generation_paths'][-1]}")
    check("krea2 edit never sends init_images or denoising_strength",
          "init_images" not in krea2_edit_request and "denoising_strength" not in krea2_edit_request,
          f"keys={sorted(krea2_edit_request)}")

    krea2_edit_crop_params = build_job_params("krea2-edit-crop")
    krea2_edit_crop_params["krea2_edit_enabled"] = True
    krea2_edit_crop_params["krea2_edit_references"] = [
        {"filename": uploaded_reference["name"], "ref_boost": 1},
    ]
    krea2_edit_crop_params["grounding_px"] = 768
    krea2_edit_crop_params["fit_mode"] = "crop"
    client.post("/api/jobs", json={"params": krea2_edit_crop_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "krea2-edit-crop"
                             and j["status"] == "completed"), None),
               20, "krea2-edit-crop job to complete")
    krea2_edit_crop_native_args = json.loads(
        find_generation_request_for_prompt("krea2-edit-crop")["prompt"]
        .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("the chosen reference fit mode reaches the runtime in the native args",
          krea2_edit_crop_native_args.get("ref_image_args")
          == "preset=krea2_edit,vlm_size=768,fit_mode=crop",
          f"ref_image_args={krea2_edit_crop_native_args.get('ref_image_args')!r}")

    uploaded_source = client.post("/api/source-image",
                                  content=base64.b64decode(ONE_PIXEL_PNG_BASE64),
                                  headers={**session_headers,
                                           "Content-Type": "image/png",
                                           "X-Filename": "img2img-source.png"}).json()
    plain_img2img_params = build_job_params("img2img-plain")
    plain_img2img_params["source_image"] = uploaded_source["name"]
    plain_img2img_params["img2img_denoise"] = 0.75
    plain_img2img_params["img2img_source_as_starting_latent"] = True
    client.post("/api/jobs", json={"params": plain_img2img_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "img2img-plain"
                             and j["status"] == "completed"), None),
               20, "img2img-plain job to complete")
    plain_img2img_request = find_generation_request_for_prompt("img2img-plain")
    check("plain img2img sends the source only as an init image",
          "init_images" in plain_img2img_request and "extra_images" not in plain_img2img_request,
          f"keys={sorted(plain_img2img_request)}")
    plain_img2img_native = json.loads(plain_img2img_request["prompt"]
                                      .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("the img2img noise multiplier travels inside the native sample args, not a top-level field",
          "img2img_noise_multiplier=1"
          in plain_img2img_native["sample_params"].get("extra_sample_args", "")
          and "extra_sample_args" not in plain_img2img_request,
          f"extra_sample_args={plain_img2img_native['sample_params'].get('extra_sample_args')!r}")

    untagged_job = next(j for j in client.get("/api/state", headers=session_headers).json()["history"]
                        if (j["params"] or {}).get("prompt") == "img2img-plain")
    check("a run that consumes no tags records an empty tag list rather than omitting it",
          untagged_job["params"].get("wd14_tags") == [],
          f"wd14_tags={untagged_job['params'].get('wd14_tags')!r}")

    tags_without_latent_params = build_job_params("source-tags-without-starting-latent")
    tags_without_latent_params["source_image"] = uploaded_source["name"]
    tags_without_latent_params["img2img_source_as_starting_latent"] = False
    tags_without_latent_params["img2img_source_tags_to_stage_one"] = True
    client.post("/api/jobs", json={"params": tags_without_latent_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "source-tags-without-starting-latent"
                             and j["status"] == "completed"), None),
               20, "source-tags-without-starting-latent job to complete")
    tags_without_latent_request = find_generation_request_for_prompt("source-tags-without-starting-latent")
    check("a source used only for tags never becomes the starting latent",
          "init_images" not in tags_without_latent_request
          and "denoising_strength" not in tags_without_latent_request,
          f"keys={sorted(tags_without_latent_request)}")
    check("a source used only for tags still runs as txt2img",
          stub_state["generation_paths"][-1] == "/sdapi/v1/txt2img",
          f"path={stub_state['generation_paths'][-1]}")

    unrouted_tag_params = build_job_params("img2img-tags-routed-nowhere")
    unrouted_tag_params["source_image"] = uploaded_source["name"]
    unrouted_tag_params["img2img_denoise"] = 0.75
    unrouted_tag_params["img2img_source_as_starting_latent"] = True
    client.post("/api/jobs", json={"params": unrouted_tag_params}, headers=session_headers)
    unrouted_tag_job = wait_until(
        lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                      if (j["params"] or {}).get("prompt") == "img2img-tags-routed-nowhere"
                      and j["status"] == "completed"), None),
        20, "img2img-tags-routed-nowhere job to complete")
    check("a source routed to no stage never loads the tagger",
          unrouted_tag_job["params"].get("wd14_tags") == [],
          "neither routing box is ticked, so nothing reads the image")
    check("a source routed to no stage leaves the prompt exactly as typed",
          find_generation_request_for_prompt("img2img-tags-routed-nowhere")["prompt"]
          .startswith("img2img-tags-routed-nowhere <sd_cpp_extra_args>"),
          "no tags may be appended when no stage asked for them")

    noisy_img2img_params = build_job_params("img2img-noise-multiplier")
    noisy_img2img_params["source_image"] = uploaded_source["name"]
    noisy_img2img_params["img2img_denoise"] = 0.75
    noisy_img2img_params["img2img_source_as_starting_latent"] = True
    noisy_img2img_params["img2img_noise_multiplier"] = 1.4
    client.post("/api/jobs", json={"params": noisy_img2img_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "img2img-noise-multiplier"
                             and j["status"] == "completed"), None),
               20, "img2img-noise-multiplier job to complete")
    noisy_img2img_native = json.loads(find_generation_request_for_prompt("img2img-noise-multiplier")["prompt"]
                                      .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("an img2img noise multiplier above 1 reaches the runtime unclamped",
          "img2img_noise_multiplier=1.4"
          in noisy_img2img_native["sample_params"].get("extra_sample_args", ""),
          f"extra_sample_args={noisy_img2img_native['sample_params'].get('extra_sample_args')!r}")

    txt2img_native = json.loads(find_generation_request_for_prompt("first")["prompt"]
                                .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("a run with no img2img source sends no img2img noise multiplier at all",
          "img2img_noise_multiplier" not in txt2img_native["sample_params"].get("extra_sample_args", ""),
          f"extra_sample_args={txt2img_native['sample_params'].get('extra_sample_args')!r}")

    referenced_source_params = build_job_params("img2img-source-as-reference")
    referenced_source_params["source_image"] = uploaded_source["name"]
    referenced_source_params["img2img_denoise"] = 0.75
    referenced_source_params["img2img_source_as_starting_latent"] = True
    referenced_source_params["img2img_use_vision_on_source"] = True
    client.post("/api/jobs", json={"params": referenced_source_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "img2img-source-as-reference"
                             and j["status"] == "completed"), None),
               20, "img2img-source-as-reference job to complete")
    referenced_source_request = find_generation_request_for_prompt("img2img-source-as-reference")
    check("the source image reaches the vision tower without becoming a DiT reference",
          referenced_source_request.get("vlm_images") == referenced_source_request.get("init_images")
          and "extra_images" not in referenced_source_request,
          f"vlm_images count={len(referenced_source_request.get('vlm_images', []))}")
    check("sending the source as a reference keeps img2img rather than becoming an edit",
          "denoising_strength" in referenced_source_request
          and stub_state["generation_paths"][-1] == "/sdapi/v1/img2img",
          f"path={stub_state['generation_paths'][-1]}")

    hires_reference_params = build_job_params("hires-lowres-reference")
    hires_reference_params["hires"] = True
    hires_reference_params["save_lowres"] = False
    hires_reference_params["hires_use_vision_on_lowres"] = True
    client.post("/api/jobs", json={"params": hires_reference_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "hires-lowres-reference"
                             and j["status"] == "completed"), None),
               30, "hires-lowres-reference job to complete")
    hires_reference_requests = find_all_generation_requests_for_prompt("hires-lowres-reference")
    check("using the low-res pass as a reference forces the low-res generation to run",
          len(hires_reference_requests) == 2,
          f"request count={len(hires_reference_requests)}")
    lowres_native_args = json.loads(hires_reference_requests[0]["prompt"]
                                    .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("the low-res pass itself carries no attachment and no hires block",
          "extra_images" not in hires_reference_requests[0]
          and "vlm_images" not in hires_reference_requests[0]
          and "hires" not in lowres_native_args,
          f"keys={sorted(hires_reference_requests[0])} native={sorted(lowres_native_args)}")
    check("the hires pass reads the low-res result for vision only, never as a DiT reference",
          len(hires_reference_requests[1].get("vlm_images", [])) == 1
          and "extra_images" not in hires_reference_requests[1],
          f"vlm_images count={len(hires_reference_requests[1].get('vlm_images', []))}")
    hires_reference_native = json.loads(hires_reference_requests[1]["prompt"]
                                        .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("a vision attachment needs no pass_to_dit workaround to stay out of the transformer",
          "ref_image_args" not in hires_reference_native,
          "its own channel is never encoded, so the reference preset does not apply to it")

    plain_hires_params = build_job_params("hires-without-reference")
    plain_hires_params["hires"] = True
    plain_hires_params["save_lowres"] = False
    client.post("/api/jobs", json={"params": plain_hires_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "hires-without-reference"
                             and j["status"] == "completed"), None),
               30, "hires-without-reference job to complete")
    plain_hires_requests = find_all_generation_requests_for_prompt("hires-without-reference")
    check("hires without the option runs once and sends no reference",
          len(plain_hires_requests) == 1 and "extra_images" not in plain_hires_requests[0],
          f"request count={len(plain_hires_requests)}")

    varying_hires_params = build_job_params("hires-varying-settings")
    varying_hires_params["hires"] = True
    varying_hires_params["save_lowres"] = False
    varying_hires_params["hires_prompt"] = "sharp focus"
    varying_hires_params["hires_prompt_mode"] = "append"
    client.post("/api/jobs", json={"params": varying_hires_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "hires-varying-settings"
                             and j["status"] == "completed"), None),
               30, "hires-varying-settings job to complete")
    varying_requests = find_all_generation_requests_for_prompt("hires-varying-settings")
    check("a hires prompt override runs the stages as two requests",
          len(varying_requests) == 2, f"request count={len(varying_requests)}")
    first_stage_native = json.loads(varying_requests[0]["prompt"]
                                    .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("the first stage renders at base resolution with no hires block",
          "hires" not in first_stage_native, f"native={sorted(first_stage_native)}")
    second_stage_native = json.loads(varying_requests[1]["prompt"]
                                     .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("the second stage upscales from the first rather than running a native hires pass",
          "hires" not in second_stage_native and "init_images" in varying_requests[1],
          f"keys={sorted(varying_requests[1])}")
    check("the second stage carries the hires prompt appended to the main prompt",
          varying_requests[1]["prompt"].startswith("hires-varying-settings, sharp focus"),
          f"prompt={varying_requests[1]['prompt'][:70]!r}")
    check("the second stage leaves the upscaled latent unnoised by the img2img path",
          "img2img_noise_multiplier=0"
          in second_stage_native["sample_params"].get("extra_sample_args", ""),
          f"extra_sample_args={second_stage_native['sample_params'].get('extra_sample_args')!r}")
    check("the second stage renders at the hires resolution",
          varying_requests[1]["width"] == varying_hires_params["hires_width"]
          and varying_requests[1]["height"] == varying_hires_params["hires_height"],
          f"{varying_requests[1]['width']}x{varying_requests[1]['height']}")

    matching_hires_params = build_job_params("hires-matching-settings")
    matching_hires_params["hires"] = True
    matching_hires_params["save_lowres"] = False
    client.post("/api/jobs", json={"params": matching_hires_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "hires-matching-settings"
                             and j["status"] == "completed"), None),
               30, "hires-matching-settings job to complete")
    matching_requests = find_all_generation_requests_for_prompt("hires-matching-settings")
    matching_native = json.loads(matching_requests[0]["prompt"]
                                 .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("unvarying hires settings keep the native single-request path",
          len(matching_requests) == 1 and matching_native.get("hires", {}).get("enabled") is True,
          f"requests={len(matching_requests)}")
    check("the hires block carries a noise multiplier, defaulting to 1",
          matching_native["hires"].get("noise_multiplier") == 1.0,
          f"hires={matching_native['hires']}")

    save_lowres_params = build_job_params("hires-save-lowres")
    save_lowres_params["hires"] = True
    save_lowres_params["save_lowres"] = True
    client.post("/api/jobs", json={"params": save_lowres_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "hires-save-lowres"
                             and j["status"] == "completed"), None),
               30, "hires-save-lowres job to complete")
    save_lowres_requests = find_all_generation_requests_for_prompt("hires-save-lowres")
    check("saving the low-res image does not render the base pass twice",
          len(save_lowres_requests) == 1,
          f"requests={len(save_lowres_requests)}; the runtime returns the pre-upscale image itself")
    save_lowres_native = json.loads(save_lowres_requests[0]["prompt"]
                                    .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("the request asks the runtime for the pre-upscale image",
          save_lowres_native["hires"].get("return_lowres_image") is True,
          f"hires={save_lowres_native.get('hires')}")
    check("a hires run that is not saving the low-res does not ask for it",
          matching_native["hires"].get("return_lowres_image") is not True,
          f"hires={matching_native.get('hires')}")

    noisy_hires_params = build_job_params("hires-noise-multiplier")
    noisy_hires_params["hires"] = True
    noisy_hires_params["save_lowres"] = False
    noisy_hires_params["hires_noise_multiplier"] = 1.6
    client.post("/api/jobs", json={"params": noisy_hires_params}, headers=session_headers)
    wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                             if (j["params"] or {}).get("prompt") == "hires-noise-multiplier"
                             and j["status"] == "completed"), None),
               30, "hires-noise-multiplier job to complete")
    noisy_requests = find_all_generation_requests_for_prompt("hires-noise-multiplier")
    noisy_native = json.loads(noisy_requests[0]["prompt"]
                              .split("<sd_cpp_extra_args>", 1)[1].split("</sd_cpp_extra_args>", 1)[0])
    check("a noise multiplier above 1 reaches the runtime unclamped",
          noisy_native["hires"].get("noise_multiplier") == 1.6,
          f"noise_multiplier={noisy_native['hires'].get('noise_multiplier')}")
    check("a noise multiplier does not by itself force a second request",
          len(noisy_requests) == 1,
          "noising the upscaled latent needs no weight reload")

    check("a job with no prompt and no tagging is refused",
          client.post("/api/jobs", json={"params": {**build_job_params(""), "prompt": ""}},
                      headers=session_headers).status_code == 400)
    check("a job with no prompt but tagging enabled is accepted",
          client.post("/api/jobs",
                      json={"params": {**build_job_params(""), "prompt": "",
                                       "img2img_source_tags_to_stage_one": True,
                                     "source_image": uploaded_source["name"]}},
                      headers=session_headers).status_code == 200)
    check("a hires replace mode with no hires prompt is refused",
          client.post("/api/jobs",
                      json={"params": {**build_job_params("replace-gate"), "hires": True,
                                       "hires_prompt": "", "hires_prompt_mode": "replace"}},
                      headers=session_headers).status_code == 400)

    served_page = client.get("/").text

    def input_tag_for_field(field_id: str) -> str:
        start = served_page.index(f'id="{field_id}"')
        return served_page[start:served_page.index(">", start)]

    login_field_tags = [input_tag_for_field("loginUser"), input_tag_for_field("loginPassword")]
    check("the login fields carry no credential signals, so password managers stay out",
          all('autocomplete="off"' in tag and 'name=' not in tag for tag in login_field_tags)
          and not any(signal in served_page
                      for signal in ('type="password"', "current-password", "new-password",
                                     'autocomplete="username"')),
          "this is a throwaway code on an ephemeral tunnel; breach warnings are pure noise")
    unguarded_fields = [field_id for field_id in ("negative_prompt", "extra_sample_args", "pag_layers")
                        if 'autocomplete="off"' not in input_tag_for_field(field_id)]
    check("generation text fields opt out of autofill so browsers stop guessing them",
          not unguarded_fields,
          f"unguarded={unguarded_fields}")

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
          == {path.name for path in krea_web.LORA_DIR.glob("*.safetensors")},
          f"listed={[e['filename'] for e in available_loras]}")

    if available_loras:
        selected_lora_filename = available_loras[0]["filename"]
        selected_lora_params = build_job_params("selected-loras")
        selected_lora_params["extra_loras"] = [{"filename": selected_lora_filename, "strength": 0.8,
                                                "use_in_main": True, "use_in_hires": True}]
        client.post("/api/jobs", json={"params": selected_lora_params}, headers=session_headers)
        wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                                 if (j["params"] or {}).get("prompt") == "selected-loras"
                                 and j["status"] == "completed"), None),
                   20, "selected-lora job to complete")
        selected_lora_request = find_generation_request_for_prompt("selected-loras")
        check("selected lora is sent with exactly its own strength and nothing else",
              selected_lora_request.get("lora")
              == [{"path": selected_lora_filename, "multiplier": 0.8, "is_high_noise": False}],
              f"lora={selected_lora_request.get('lora')}")

        unselected_params = build_job_params("no-selected-loras")
        unselected_params["extra_loras"] = []
        client.post("/api/jobs", json={"params": unselected_params}, headers=session_headers)
        wait_until(lambda: next((j for j in client.get("/api/state", headers=session_headers).json()["history"]
                                 if (j["params"] or {}).get("prompt") == "no-selected-loras"
                                 and j["status"] == "completed"), None),
                   20, "no-selected-lora job to complete")
        unselected_request = find_generation_request_for_prompt("no-selected-loras")
        check("unselected loras are not sent to the backend",
              "lora" not in unselected_request, f"keys={sorted(unselected_request)}")

    check("unloading models needs authentication",
          client.post("/api/unload-models").status_code == 401)

    recorded_subprocess_calls: list[list[str]] = []
    real_subprocess_run = krea_web.subprocess.run
    krea_web.subprocess.run = lambda command, **kwargs: recorded_subprocess_calls.append(list(command))
    try:
        unload_response = client.post("/api/unload-models", headers=session_headers)
    finally:
        krea_web.subprocess.run = real_subprocess_run
    check("unloading models stops the server so its VRAM is released",
          unload_response.status_code == 200
          and any(call[-1] == "stop" and call[0].endswith("krea-server.sh")
                  for call in recorded_subprocess_calls),
          f"calls={recorded_subprocess_calls}")
    check("unloading forgets the loaded checkpoint so the next job reloads it",
          app.state.queue_manager.loaded_checkpoint == "" and unload_response.json()["model"] == "",
          f"loaded_checkpoint={app.state.queue_manager.loaded_checkpoint!r}")

    app.state.queue_manager.current = "pretend-running-job"
    try:
        busy_unload_status = client.post("/api/unload-models", headers=session_headers).status_code
    finally:
        app.state.queue_manager.current = None
    check("unloading is refused while a job is running",
          busy_unload_status == 409,
          "pulling weights out from under a running generation would fail it")

    check("logout invalidates the token immediately",
          client.post("/api/logout", headers=session_headers).status_code == 200
          and client.get("/api/state", headers=session_headers).status_code == 401)

    stub_server.shutdown()
    print()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("all queue control checks passed")
    return 0


def run_main_with_a_disposable_output_directory() -> int:
    """Keep the suite's uploads and stub renders out of the real outputs folder.

    The queue checks post through the real upload and save routes, so without
    this every run would leave stub images behind for the user to sift out.
    """
    original_output_dir = krea_web.OUTPUT_DIR
    with tempfile.TemporaryDirectory() as disposable_output_directory:
        krea_web.OUTPUT_DIR = Path(disposable_output_directory)
        try:
            return main()
        finally:
            krea_web.OUTPUT_DIR = original_output_dir


if __name__ == "__main__":
    raise SystemExit(run_main_with_a_disposable_output_directory())
