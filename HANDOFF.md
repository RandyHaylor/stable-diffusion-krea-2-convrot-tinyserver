# Handoff — 2026-08-29

Branch: `feature/krea2-identity-edit-dual-conditioning`

Everything below is committed. The web UI and model server were both restarted
against this build and are running.

---

## 1. What shipped today

### Per-stage LoRA selection

Each LoRA now has independent **main** and **hires** checkboxes, so a LoRA can be
main-only, hires-only, or both. `select_loras_for_stage` filters on the stage's
own tick instead of returning everything for main. The tests for this were
already written and failing before the change; they pass now.

### img2img noise multiplier

`img2img_noise_add` was inert. It was written to `payload["extra_sample_args"]`,
a **top-level** A1111 body field that `examples/server/` never reads — the only
channel is the `<sd_cpp_extra_args>` blob inside the prompt, under
`sample_params` (`routes_sdapi.cpp:144`, `common.cpp:2012`). The blob was also
serialized before those lines ran. The sampler defaulted to `noise_add = 1.f`,
so every setting behaved identically.

Renamed to `img2img_noise_multiplier`, default 1, uncapped above 1, built into
`sample_params` before the prompt is serialized. It scales only the sigma used to
build the start latent, independently of the schedule the sampler walks.

**Caveat for anyone tuning it:** the flow denoiser mixes as
`source·(1−σ) + noise·σ`. Above σ=1 the source coefficient goes negative — that is
a different regime, not simply more noise. `sigmas[0]` is already ≈ the denoise
strength, so a multiplier of ~1.33 at denoise 0.75 already crosses it.

### WD14 tagging gate

`img2img_wd14_tag` and the per-reference `tag` boxes select *which* images get
read. A stage then has to consume them: `main_append_wd14_tags` for the main
prompt, or a hires tag source of `reference_images`. Previously the gate ignored
the selection boxes entirely, so ticking "Tag source with WD14" alone did nothing
— verified against the user's own A/B in PNG metadata, where ticked and unticked
produced byte-identical prompts.

New `reference_tags_are_consumed()` in `prompt_composition.py`. When nothing
consumes the tags the boxes show an orange `(unused)` and the tagger is never
loaded. The tags actually used are recorded in `p["wd14_tags"]`, which reaches
both the output panel and the saved PNG.

### Save low-res no longer renders the base pass twice

This was the day's real bug. `run_first_stage_separately` was true when
`save_lowres` was set, but the second request only *consumed* that first stage
when `settings_vary` was true. With `save_lowres` on and settings matching, the
base resolution was rendered twice — confirmed in a real log: 65.71s then a
duplicate 52.92s at the same resolution, same seed, same LoRAs.

Fixed in the runtime rather than by re-plumbing the app: `sd_hires_params_t`
gained `return_lowres_image`, and `generate_image` decodes `final_latents` before
the hires pass replaces them, returning `[low-res..., hires...]`. Cost is one VAE
decode instead of a full second sampling run. `save_lowres` no longer forces a
split; only vision-on-lowres and stage-one tags do, because those genuinely need
the file on disk before the next request is built.

### Other

- Hires reference size dropdown (`auto|384…1024`), an edge length sent as an N×N
  `vae_input_max_pixels` budget, applied to the hires request only.
- Reference fidelity other than exactly 1 now shows `(extra VRAM)` — see §3.
- Default sampler `er_sde` → `euler`.
- "Use vision on low-res" renamed to "Get vision data from lowres for hires pass";
  its tooltip's DiT-reference VRAM warning was stale and is gone.
- **Unload models** button (`POST /api/unload-models`) — stops the model server so
  the GPU gets its memory back. Verified live: 17,618 MiB → 172 MiB. Idempotent,
  409 while a job runs, and it clears `loaded_checkpoint` so the next job reloads.
- Reference VAE encode is skipped when references never reach the DiT
  (`pass_to_dit=false`), saving ~2.3s on vision-only runs.
- `test_krea_web_queue_controls.py` runs under a `TemporaryDirectory`, so it no
  longer writes 1×1 PNGs into `outputs/`.

---

## 2. The open problem: OOM on krea2 edit at 1248×1824

**Unresolved. Do not assume a cause.** Two confident explanations were offered
during the session and both were disproved by measurement.

Measured facts:

- The failing allocation is ~4.4–4.6 GB for the diffusion compute buffer.
- At 1248×1824 the target is 78×114 = 8892 tokens and a full-budget reference is
  ~4081 tokens, so the sequence is ~13,000. The buffer scales roughly with its
  square.
- The reference does **not** grow with the hires target — 3952 tokens at
  832×1216, 4081 at 1248×1824. The ~1M px cap holds.
- Params report 17,707 MB but a clean process sits at 17,440 MB after a small
  generation, so that figure is a registry total, not residency.

**Disproved, with the experiment that killed each:**

1. *"A failed allocation leaves the pool grown; one OOM poisons the process."*
   Forced an OOM at 3072×3072 on a healthy process, then re-ran the 1248×1824
   that had just succeeded. It succeeded again; VRAM went 18,432 → 18,430 →
   18,430. `ggml_extend.hpp:2235` already calls `free_compute_buffer()` on the
   reserve failure.
2. *"The allocator is a high-water mark that never returns memory."* The baseline
   dropped 19,854 → 18,432 after a successful run.

Also checked: the VMM pool (`ggml-cuda.cu:536-683`) only releases in its
destructor and `ggml-cuda.h` exposes no trim API, so there is nothing to call
even if it had been the cause.

Still unexplained: the user reports this configuration worked before recent
changes. That has **not** been tested. The clean test is now easy — press
**Unload models**, then run the exact config on freshly loaded weights.

---

## 3. Reference fidelity costs VRAM

`krea2.hpp:901` allocates a dense `pos_len × pos_len` f32 tensor in the compute
buffer whenever any reference fidelity `!= 1.0f` (`:885-887`) — in **either**
direction, so 0.3 and 0.5 both pay it. At pos_len ≈ 13,000 that is ~675 MiB.
Exactly 1 skips the mask entirely. The UI marks non-1 fidelity `(extra VRAM)`.

The mask does **not** disable flash attention — it is passed into
`ggml_ext_attention_ext` with `flash_attn_enabled` (`krea2.hpp:232-240`).

---

## 4. Known-unfinished

1. **The unload button's recovery path is not fully verified.** A queued job only
   restarts the server via `switch_checkpoint` when `checkpoint` is set. The
   queue-control tests send no checkpoint at all and are accepted, so the empty
   branch is reachable through the API; whether the browser can produce it was
   not tested. An `ensure_model_server_running()` guard was proposed and not
   written.
2. **The hires stage still sends edit references to the DiT at full strength.**
   `krea_web.py` skips the vision-only `pass_to_dit=false` path for edit mode by
   design, and the hires request inherits it. The reference-size dropdown is the
   mitigation; it has not been measured against a real OOM.
3. **Nothing here has been A/B'd for image quality** — the hires reference size,
   the noise multiplier, and vision-only conditioning are all untested visually.
4. `krea2_ostris_edit` is the standing default preset whenever `ref_image_args`
   is absent (`stable-diffusion.cpp:3207-3211`); it logs on every generation.
   Whether its fields can affect a zero-reference run was not checked.

---

## 5. Conventions that bit us

- Do not commit inside `vendor/`. Vendor changes live in
  `patches/stable-diffusion-cpp-int8-rowwise.patch`, regenerated with
  `git -C vendor/stable-diffusion.cpp diff > patches/...`.
- Rebuild after vendor edits:
  `cmake --build vendor/stable-diffusion.cpp/build --config Release -j 16 --target sd-server`
- `krea_web.py` changes need a web restart; `web/index.html` is read per request.
- **Do not run destructive operations against the live server while the user is
  working.** Restarts, forced OOMs, and unload calls during this session
  interrupted real work — one of them left the server stopped and produced a
  connection-refused that looked like a product bug.
