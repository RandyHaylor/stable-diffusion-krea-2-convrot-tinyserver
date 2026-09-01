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
    - `scripts/test_tiled_diffusion.py`
    - `scripts/test_source_image_sizing.py`
    - `scripts/test_executed_steps.py`
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

An existing image can be refined with no first stage sampled at all
    - `img2img_source_replaces_first_stage` sends the source to the hires stage;
      `renders_hires_from_existing_source()` in `hires_staging.py` decides
    - It does NOT require tiling. The hires pass always continues the main pass's
      latent (`stable-diffusion.cpp:6272`), so the source is simply encoded as
      that latent with the first stage's strength forced to 0
    - The stage one prompt STILL conditions the request. The UI dims only what is
      genuinely unused, the stage one steps and the img2img denoise
    - The source is the USER'S file and is never unlinked

Tiled diffusion is the only tiling, and it is chosen as a GRID
    - `tiled_diffusion = on` splits the latent into overlapping tiles, denoises
      each with the real model and fuses them under a raised cosine EVERY step
    - One request, no VAE round trip, no per-tile requests. Because the fusion is
      per step rather than per finished tile, neighbours cannot drift apart
    - The UI picks `tiled_diffusion_grid` (1x1 .. 3x3), NOT a tile size. Tile
      count is what the ghosting guidance is keyed on and what sets the cost of a
      step, so it is the number worth choosing
    - Tile size is DERIVED per axis: `size = (length + overlap * (count - 1)) /
      count`, rounded up to a whole latent cell. A grid over a canvas that is not
      square gives tiles that are not square
    - The runtime therefore carries `tiled_diffusion_tile_width` and
      `tiled_diffusion_tile_height` separately. One scalar could not express it,
      and forcing square tiles onto a portrait canvas spent most of their area on
      redundant overlap
    - VERIFIED on the GPU: a 2x2 grid over 1248x1824 logs `latent tile 86x122`
      and `4 tiles per step (2 x 2) over a 156x228 latent`
    - The notice reports the DERIVED tile size and the overlap that actually
      results, since rounding moves both away from the requested overlap
    - Decided by the LARGEST canvas the request reaches, because one request
      renders both passes off the same sample args. The runtime denoises a pass
      whole when it fits in one tile, so the main pass is untouched while the
      hires pass tiles
    - `scripts/tiled_diffusion.py` is pure and tested; the UI is in PIXELS and
      that module divides by 8

A source sent to the hires stage sizes itself, ignoring the main resolution
    - `source_pixel_budget_edge` is an AREA budget: 1024 means 1024x1024 pixels'
      worth, and the source keeps its own aspect inside it
    - `source_size_increment` rounds each side DOWN to a multiple, which makes the
      fit a small centre crop rather than a squash. Nothing is ever stretched
    - An increment of 1 can land off the VAE's factor of 8. It is offered anyway;
      64 is the default
    - The main pass resolution is NOT consulted. It is dimmed in the UI along with
      the stage one steps and the img2img denoise, and the size the source will
      actually be encoded at is reported instead
    - `scripts/source_image_sizing.py` is pure and tested

The steps setting means steps EXECUTED, and the app scales to make that true
    - The runtime builds a schedule of the requested length and runs only its
      tail: `t_enc = int(sample_steps * strength)`, at `stable-diffusion.cpp:5264`
      for img2img and `:5990` for the in-request hires
    - Left alone, 8 steps at denoise 0.4 spends THREE evaluations
    - `scheduled_steps_for_executed_steps()` scales the count before sending, so
      8 at 0.4 sends 20 and the runtime runs 8. The denoise still picks the
      starting sigma; only the count changes
    - Applied at all three sites that carry a denoise: the img2img main pass, the
      in-request hires stage, and each hires tile request
    - VERIFIED on the GPU against the runtime's own `target t_enc` line at
      denoise 0.25, 0.4, 0.6 and 0.75
    - NOT applied when `strength_as_noise_level` appears in the free-form sample
      args: that branch derives the count from the sigma, so scaling would miss
    - Every measurement taken before this was understood ran 2-3 steps per
      tile while being labelled 8

RoPE offsets per tile measured WORSE, and are off by default
    - `tiled_diffusion_rope_offset = on` gives each tile position ids at its
      true canvas place instead of every tile claiming the origin
    - The theory was that fusing predictions made under contradictory coordinate
      assumptions is what hurt. It did not survive testing
    - At 2-3 effective steps the offsets looked like a clear win. At 8 real steps
      the plain origin-labelled tiles read better for coherence, so the gain
      was the step count, not the positions
    - Kept behind the setting for comparison, never on by default

Tiles must overlap, and an exact doubling does not overlap on its own
    - Two 1024px tiles reach 2048px only by abutting, leaving nothing to blend
    - `tile_start_positions_covering_length()` spreads a computed count evenly
      instead of marching at the stride, so overlap is uniform
    - An exact 2x therefore needs THREE tiles per axis, not two

Ghosting guidance is keyed on tile count, and is thin
    - 0.6 up to 4 tiles, 0.35 above. Measured against independently refined pixel
      tiles, where two neighbours could invent different content
    - Tiled diffusion fuses every step, so neighbours cannot diverge that way and
      the guidance is likely conservative here. NOT re-measured against it
    - An earlier version keyed it on upscale factor. That did not survive testing

The pixel-tiling hires path was removed after losing its comparison
    - It rendered the first stage alone, decoded it, resampled it and repainted it
      as overlapping tiles, one request each, blending them at the end
    - Two attempts to save it both lost: anchoring each tile to its finished
      neighbours, and the Ultimate SD Upscale paradigm proper (exact cells,
      padding as context, mask blur) in `spike_a_ultimate_sd_upscale.py`
    - Tiled diffusion beat both, so `hires_tiling.py`, `tiled_refine.py`, their
      spikes and their three test suites were deleted rather than left as a second
      way to do the same job
    - Judge any tiling by the seam crops, not by seam metrics: a ghost is a
      smeared low-contrast region, and blur reads as a SHALLOW gradient, so a
      seam-steepness metric rates a ghosted result BETTER

=== Response summary ===

- What This Is
- Running The Server
- Change Rules
- Architecture Gotchas
