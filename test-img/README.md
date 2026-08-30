# Tiled hires test images

Evidence for the tiled hires pass: a 1536x2304 built from four 832x1216 tiles
overlapping 128px, on a GPU that renders 832x1216 in one pass. Turbo LoRA,
6 step schedule, hires denoise 0.6, seed 12345.

Look at them in this order.

| image | what it shows |
|---|---|
| `tiled-hires-0-base-832x1216.png` | the first stage, rendered in one pass |
| `tiled-hires-1-lanczos-only-1536x2304.png` | the same image resampled up, nothing repainted. The detail baseline. |
| `tiled-hires-2-unblended-seams-1536x2304.png` | tiles pasted with no cross-fade, so the seams are visible |
| `tiled-hires-3-blended-1536x2304.png` | the result |
| `tiled-hires-8-through-app-1536x2304.png` | the same thing produced by the app's own hires path rather than the spike script |

Seam crops, all the same region at full resolution:

| image | what it shows |
|---|---|
| `tiled-hires-4-crop-seam-lanczos.png` | resample only, for comparison |
| `tiled-hires-5-crop-seam-unblended.png` | the hard seam: the railing steps and a rectangular block is obvious |
| `tiled-hires-6-crop-seam-blended.png` | seam gone, but the sleeve ghosts where the two tiles disagreed about the coat edge |
| `tiled-hires-7-crop-vseam-blended-clean.png` | the vertical seam through flat fabric, with no artefact |

## Measured

Detail energy, mean absolute second derivative of luminance:

| | detail | vs resample |
|---|---|---|
| resampled canvas | 2.601 | 1.000x |
| tiles pasted | 4.577 | 1.760x |
| tiles blended | 4.489 | 1.726x |

Seam visibility at the interior boundaries, as a multiple of the image's own
median row-to-row change:

| | worst seam row | rank among 2303 rows |
|---|---|---|
| pasted | 26.10x | 1st steepest |
| blended | 1.13x | 986th |

Inter-tile disagreement over the shared regions, having started from identical
pixels, which is what the ghosting in crop 6 is made of:

| tile pair | mean | 99th percentile |
|---|---|---|
| left / right | 6.93 | 56 |
| top / bottom | 24.61 | 115 |

Crops 6 and 7 are the same blend on the same run: 7 is clean because it crosses
flat fabric, 6 ghosts because it crosses the railing, the coat hem and a hand.
Ghosting tracks structural disagreement between independently refined tiles, not
the blend.

## Anchoring fixes the ghosting

Writing each finished tile back before the next is cut, so a tile starts from its
neighbour's refined pixels, removes the ghost entirely. Same seeds, same four
passes, same cost.

| image | what it shows |
|---|---|
| `tiled-hires-9-crop-seam-independent-ghosting.png` | every tile cut from the same resampled canvas: the sleeve is translucent over the railing |
| `tiled-hires-10-crop-seam-anchored-clean.png` | the same region anchored: railing continuous, grain crisp, no ghost |
| `tiled-hires-11-anchored-1536x2304.png` | the full anchored result |

A warning about the metric, kept because it nearly produced the wrong decision.
Seam steepness rated the ghosted version *better*:

| | seam row | rank among 2303 rows | detail |
|---|---|---|---|
| independent | 1.13x median | 986th | 4.489 |
| anchored | 2.08x median | 542nd | 4.549 |

A ghost is a smeared, low-contrast region, and blur reads as a low gradient, so
the artefact scored well on a metric meant to detect artefacts. The anchored
version's higher number is the railing edge being preserved rather than averaged
away. Crops 9 and 10 are why anchoring is the default; the numbers alone argued
the opposite.
