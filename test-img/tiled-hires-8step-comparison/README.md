# Tiled hires at 8 steps, with the identity-edit LoRA

Every image here shares one setup. The only thing that differs between a pair is
named in its filename.

    subject     bookshelf wall: straight shelves the full width, evenly spaced
                spines, checkered floor with straight grout lines
    sampler     euler / discrete
    steps       8 main, 8 hires
    cfg         1.0, flow shift 1.15
    LoRAs       krea2_raw_to_turbo_r256 at 0.6, krea2_identity_edit_v1_2 at 1.0
    seed        4242
    first stage 832x1216, one pass
    target      2432x3648, a 2.96x upscale
    tiling      4x4 = 16 tiles of 832x1216, 128px overlap, anchored, single hop
    denoise     0.6

The subject was chosen because the earlier one could not be read: a person's body
crossed the railing, so an occlusion was indistinguishable from a seam. Here a
shelf line that steps, or a grout line that kinks, can only be a seam.

| image | what it is |
|---|---|
| `00-first-stage-832x1216-8steps.png` | the one-pass first stage every tiled run starts from |
| `01-tiled-3x-2432x3648-tile-vision-OFF.png` | tiled result, tiles conditioned on the prompt only |
| `02-tiled-3x-2432x3648-tile-vision-ON.png` | tiled result, each tile also read by the vision tower |
| `03..04-crop-row-seam-1to1-*` | the same horizontal tile boundary at 1:1, both modes |
| `05..06-crop-column-seam-1to1-*` | the same vertical tile boundary at 1:1, both modes |

The vertical divider visible in the crops is a real bookshelf upright present in
the first stage, not a seam.

## Measured

| | detail energy | wall clock |
|---|---|---|
| tile vision off | 3.529 | 260s |
| tile vision on | 3.516 | 320s |

Detail energy is mean absolute second derivative of luminance, a sharpness proxy.

## What these show

**8 steps and the identity-edit LoRA matter.** The same 2432x3648 target at 6
steps, turbo at 1.0 and no identity-edit LoRA measured 2.695 detail and ghosted
visibly. Here it measures 3.529, a 31% gain, and the seams read clean. Effective
steps per tile is `int(steps * denoise)`, so 6 steps at 0.6 denoise spent 3 steps
per tile and 8 spends 4.

**Tile vision showed no gain in this test, and cost 23% more time.** That is a
result about this subject, not a verdict on the feature: a bookshelf is
homogeneous, so the prompt already describes every tile accurately and telling a
tile "you are shelving" adds nothing it did not have. The case it was meant for
is a heterogeneous image, where one tile is all floor and another all sky and the
whole-image prompt fits neither. That has not been tested yet.

**Not tested here: the tile vision weight.** These runs used a neutral weight, so
the weighting path was inert. The weight itself is confirmed working against the
running server, which logs
`vlm_image_token_weight 0.4 on vision image 1 (144 token(s) from index 40)`
when it scales, but no image in this folder varies it.
