# Notes and questions — per-stage routing work

Written at the end of the unattended run. Everything below is committed and
pushed to `main`; the five increments are `ee78b58`, `d233540`, `81893d6`,
`e5d9e05`, `39d8456`, plus this doc.

---

## 1. Machine state — read this first

**The running processes are both stale.**

- `sd-server` is running the binary from *before* the `vlm_images` change. The
  app now sends a `vlm_images` field it will silently ignore, so vision on the
  img2img source and on the low-res output will do nothing until it restarts.
- The web app is running the `krea_web.py` from before all of today's changes.

I built the new binary and verified it links, but I did **not** restart either
process — you had asked me to stop touching the live server, and I did not want
to interrupt you again. A `./restart-krea.sh` in its own terminal picks up both.

**I have not run a single real generation against any of this.** Every check
below is unit/integration tests against the stub backend plus source reading. The
whole feature set is unproven on the GPU.

---

## 2. Questions I need answered

### 2a. Should `use as starting latent` be allowed with Krea2 Edit?

Currently **refused**: with edit mode on, the source is never the starting
latent. My only basis is our own `krea2_edit_request.py` docstring saying the
edit target "starts as pure noise" and "must never carry init_images". That is
*our* stated design intent — I could not verify from upstream or from the LoRA
whether edit actually breaks with a non-noise start.

You said "nothing about the process makes krea2edit and img2img mutually
exclusive". If you want to test that, say so and I will unblock it. Tags and
vision from the source already coexist with edit mode.

### 2b. Krea2 Edit reference vision is still not per-stage

Your spec wanted per-image vision routing for edit references too. I did not do
it, deliberately, because there are two ways and they differ in behaviour:

- put a copy of the reference in `vlm_images` as well → the tower encodes it
  twice for a request where both are wanted, wasting time
- make `pass_to_vlm` per-reference like `ref_boost` → a vendor change, and it
  splits a coupling upstream deliberately made

Edit references currently reach the tower through the preset, both stages, as
upstream intends. Which way do you want it?

### 2c. `(unused)` markers are gone

They no longer have meaning — a routing box *is* the consumption decision, so
nothing can be ticked-but-ignored. If you liked the warning for some other case,
tell me what case.

---

## 3. What I could not verify

- **Whether any of this fixes the 1248×1824 OOM.** The mechanism is addressed:
  the low-res image can no longer become a second DiT reference, so the
  17,050-token sequence from your log should not recur. But I have not run it.
- **Whether the `vlm_images` prompt ordering is right.** The Krea2 conditioner
  now walks references first, then vision-only images, numbering them
  `Picture 1..N` in that combined order. The model was trained on some specific
  arrangement and I do not know what it was. With edit mode and img2img sources
  mutually exclusive for the *latent*, both lists can now be non-empty at once —
  that combination has never been run.
- **Any image-quality effect** of the new routing, the noise multiplier, or the
  hires reference size. None of it has been A/B'd.

---

## 4. Smaller things I noticed and left alone

- `krea_web.py` still has a `build_vision_only_ref_image_args` helper in
  `krea2_edit_request.py` with tests, now unused by the app. It is the old
  `pass_to_dit=false` workaround. Harmless, but dead.
- The `hires_use_vision_on_lowres` param keeps its old name while its UI label
  says "Get vision data from lowres for hires pass". Renaming it would orphan the
  params stored in existing output PNGs.
- The vendor's `krea2_ostris_edit` preset is still the standing default whenever
  `ref_image_args` is absent, and logs on every generation. Whether its fields
  affect a zero-reference run is still unchecked.

---

## 5. The `auto-resume-on-stop` skill

I told you earlier this skill "doesn't exist". That was wrong — it is at
`~/.claude/skills/auto-resume-on-stop`. I had checked the available-skills
listing and `~/.claude/offloaded-skills/index.md` and reported absence from those
two places as absence in general.

It is installed and armed for this project via `stop-settings.json` in the
session cwd (the parent of this repo, not the repo itself — the hook reads the
session's cwd). I have set `auto-resume` to `false` now that the plan is done and
these questions need you.
