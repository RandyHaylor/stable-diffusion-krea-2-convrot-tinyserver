# CLAUDE.md

=== What This Is ===

Local Krea 2 image generation, RTX 3090 Ti, 24 GB
    - Vendored `stable-diffusion.cpp` runtime, CUDA, INT8 ConvRot DiT
    - Python FastAPI web UI in front of it
    - Two processes
        * `sd-server` on 127.0.0.1:1234, holds the weights
        * `krea_web.py` on 127.0.0.1:7860, queue + UI + Cloudflare tunnel

Layout
    - `scripts/` — web app and its isolated modules, plus their tests
    - `web/index.html` — entire UI, single file
    - `vendor/stable-diffusion.cpp/` — gitignored upstream checkout
    - `patches/stable-diffusion-cpp-int8-rowwise.patch` — all vendor changes
    - `models/`, `outputs/` — gitignored

=== Running The Server ===

Launch the server in its own terminal window
    - NOT tied to an agent session
    - Spawn a NEW process, detached from the agent's shell
    - Run `./restart-krea.sh`
        * Stops web UI + tunnel, stops model server, starts both
        * Ends in `tail -F` on the server log, so it never returns
        * That is why it needs its own window

Why it must be independent
    - An agent-owned shell dies with the turn, taking the server with it
    - Killing or unloading mid-session interrupts the user's real work
    - Do NOT restart, unload, or fire test generations against a live server
        * Ask the user to run it instead

Reading the logs
    - `.runtime/krea-server.log` — sd-server, generation detail
    - `.runtime/krea-web.log` — web app, WD14 tagging lines live here
    - `.runtime/krea-web-credentials` — user, password, tunnel URL

=== Change Rules ===

Never commit inside `vendor/`
    - Its `AGENTS.md` forbids agent commits
    - Regenerate instead
        * `git -C vendor/stable-diffusion.cpp diff > patches/stable-diffusion-cpp-int8-rowwise.patch`

Rebuild after any vendor edit
    - `cmake --build vendor/stable-diffusion.cpp/build --config Release -j 16 --target sd-server`
    - New binary needs a server restart to take effect

Restart scope
    - `krea_web.py` changed -> web restart required
    - `web/index.html` changed -> live immediately, read per request

Tests first, then the code they cover
    - `scripts/test_hires_staging.py`
    - `scripts/test_prompt_composition.py`
    - `scripts/test_krea2_edit_request.py`
    - `scripts/test_source_upload_and_seed.py`
    - `scripts/test_krea_web_queue_controls.py`
    - `scripts/test_wd14_tagging.py`
    - Plain `python3 <file>`, no pytest
    - Point `krea_web.OUTPUT_DIR` at a temp dir in any suite that saves images

=== Architecture Gotchas ===

Native args reach the runtime inside the prompt, not the request body
    - `<sd_cpp_extra_args>{json}</sd_cpp_extra_args>` appended to `prompt`
    - Top-level body keys the SDAPI does not name are silently DROPPED
    - `extra_sample_args` lives under `sample_params` in that blob

One request carries one LoRA set, one prompt, one `ref_image_args`
    - Any per-stage variation forces the hires stage into its own request
    - `hires_settings_vary_from_main()` decides
    - Costs a weight reload and the in-request latent continuity

Krea2 Edit overrides img2img
    - Edit mode enabled -> the img2img source image is discarded

Reference tokens share the target's attention sequence
    - Compute buffer grows with the square of the total
    - Reference fidelity other than exactly 1.0 adds a dense `pos_len x pos_len` f32 mask

Vision tower is indexed at startup, NOT resident
    - `loading llm vision` is `model_loader` reading the file, not a VRAM upload
    - `total params memory size` is a registry total, NOT residency

=== Response summary ===

- What This Is
- Running The Server
- Change Rules
- Architecture Gotchas
