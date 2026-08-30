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
    - `scripts/test_tiled_refine.py`
    - `scripts/test_hires_tiling.py`
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

Tiled hires is a separate path, not a setting on the normal one
    - `hires_tiling = on` -> `renders_hires_by_tiling()` routes around the
      in-request hires entirely
    - First stage runs alone, is decoded and resampled to the full target, then
      repainted as overlapping tiles, one request each
    - Tile size IS the main pass resolution, the one size the job proved fits
    - Trades the in-request latent continuity for reach; declines to engage when
      the target needs a single tile
    - `scripts/tiled_refine.py` holds the geometry and blending, pure and tested

Tiles must overlap, and an exact doubling does not overlap on its own
    - Two 832px tiles reach 1664px only by abutting, leaving nothing to blend
    - `tile_start_positions_covering_length()` spreads a computed count evenly
      instead of marching at the stride, so overlap is uniform and never under
      the minimum
    - An exact 2x therefore needs THREE tiles per axis, not two

Blending removes a seam step but cannot reconcile content
    - Independently refined neighbours disagree; cross-fading a disagreement
      makes a translucent ghost, not a clean join
    - `hires_tile_source = anchored` (the default) writes each finished tile back
      before the next is cut, so a tile starts from its neighbour's own pixels
    - Seam-steepness metrics RATE A GHOST WELL: blur reads as a shallow
      gradient. Judge tiling by the seam crops, not by the numbers.

Usable hires denoise falls as the tiling upscale factor rises
    - A blurrier resample gives each tile more freedom to invent differently
    - Measured: 2x held together at 0.6; 3x ghosted at 0.6, clean at 0.35
    - `recommended_maximum_hires_denoise_for_tiling()` is guidance from those
      three points, surfaced in the UI notice, enforced nowhere

=== Response summary ===

- What This Is
- Running The Server
- Change Rules
- Architecture Gotchas
