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

One request carries one prompt and one `ref_image_args`, shared by both stages
    - Per-stage variation makes the job PAUSE between its passes, not split into
      two requests; `hires_settings_vary_from_main()` decides
    - The latent is held across the pause, so continuity is never given up
    - LoRAs ARE per stage: the hires selection rides the pause on
      `sd_hires_stage_input_t.loras` and is applied between the passes
    - An empty hires selection unloads the main pass's LoRAs rather than letting
      them carry over, which is what `applies_own_loras` exists to express
    - An identical selection is detected and skipped, so the common case reloads
      nothing

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

A tiled hires tile can be read by the vision tower, like the img2img source
    - `hires_tile_vision = on` sends the tile's own starting pixels as
      `vlm_images` too, so the repaint is conditioned on this tile's content
      rather than on a prompt written for the whole image
    - `hires_tile_vision_weight` scales those tokens via `vlm_image_token_weight`
    - ON by default. A tile is conditioned on the prompt for the WHOLE image, so
      a tile of nothing but floor is told the scene has a window and an armchair,
      and at denoise 0.6 it paints them in as a reflection
    - Measured: no gain on a homogeneous subject at 23% more time, but on a
      heterogeneous one it is the difference between floor staying floor and
      becoming a mirrored copy of the room
      (`test-img/tile-vision-heterogeneous-comparison/`)
    - Detail-energy metrics are BLIND to this: 2.027 against 2.051 for a
      difference obvious by eye. An invented reflection is as sharp as a floor.

Blending removes a seam step but cannot reconcile content
    - Independently refined neighbours disagree; cross-fading a disagreement
      makes a translucent ghost, not a clean join
    - `hires_tile_source = anchored` (the default) writes each finished tile back
      before the next is cut, so a tile starts from its neighbour's own pixels
    - Seam-steepness metrics RATE A GHOST WELL: blur reads as a shallow
      gradient. Judge tiling by the seam crops, not by the numbers.

Effective steps per tile is `int(steps * denoise)`, and too few causes seams
    - `stable-diffusion.cpp:5073`; 6 steps at denoise 0.6 spends THREE
    - Too few steps means each tile guesses coarsely at whatever crosses a shared
      band, and two coarse guesses from slightly different pixels diverge
    - 8 steps with `krea2_raw_to_turbo_r256` at 0.6 and
      `krea2_identity_edit_v1_2` at 1.0 measured 31% more detail than 6 steps
      with turbo alone at 1.0, and read clean where the old settings ghosted
    - The identity-edit LoRA is for preserving an init, which is exactly what a
      tile anchored on its neighbour needs

Ghosting guidance is keyed on tile count, and is thin
    - `recommended_maximum_hires_denoise_for_tiling()`: 0.6 up to 4 tiles, 0.35
      above, surfaced in the UI notice, enforced nowhere
    - An earlier version keyed it on upscale factor. That did not survive
      testing: two hops each within a doubling still ghosted, so factor is not
      the variable
    - VALIDATED on a heterogeneous subject at 8 steps with the identity-edit
      LoRA: a 16 tile grid at 0.6 invented a reflected room into a plain floor,
      the same grid at the recommended 0.35 came back clean
    - The cost is softness, not nothing: detail 1.704 at 0.35 against 2.051 at
      0.6. A less crowded grid can afford the higher value.

Judge tiling on the seam crops, not on seam metrics
    - A ghost is a smeared low-contrast region, and blur reads as a SHALLOW
      gradient, so a seam-steepness metric rates a ghosted result BETTER
    - Measured: ghosted 1.13x median at the seam row, clean 2.08x
    - A subject whose own content crosses a seam is unreadable as a test. Use
      straight lines running the full width and height.

=== Response summary ===

- What This Is
- Running The Server
- Change Rules
- Architecture Gotchas
