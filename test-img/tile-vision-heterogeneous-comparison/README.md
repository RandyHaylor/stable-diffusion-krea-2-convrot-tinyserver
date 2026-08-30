# Tile vision on a subject whose tiles differ

This is the comparison that decided `hires_tile_vision` should default to on.
Look at images 01 and 02 first; they are the whole result.

    subject     interior: dark hardwood floor foreground, bright window with a
                city skyline, red armchair, brass lamp, books, patterned rug
    sampler     euler / discrete, 8 steps main and hires, cfg 1.0, flow 1.15
    LoRAs       krea2_raw_to_turbo_r256 at 0.6, krea2_identity_edit_v1_2 at 1.0
    seed        909
    first stage 832x1216, one pass
    target      2432x3648, 4x4 = 16 tiles of 832x1216, 128px overlap
    tiling      anchored, single hop, denoise 0.6
    difference  tile vision off in one run, on in the other. Nothing else.

The subject was chosen for regions that are unalike. A tile cut from the bare
floor contains no window, no armchair, no lamp and no books, yet the prompt tells
it about all four.

| image | what it is |
|---|---|
| `00-first-stage-832x1216.png` | the one-pass base. Note the floor: plain dark planks with a faint sheen, no mirrored room. |
| `01-crop-FLOOR-...-OFF-invents-a-reflected-room.png` | that floor after a tiled pass with no tile vision |
| `02-crop-FLOOR-...-ON-stays-floor.png` | the same region with tile vision on |
| `03..04-crop-upper-1to1-*` | a region the prompt already describes well, both modes |
| `05..06-full-2432x3648-*` | the whole results |

## What 01 and 02 show

Without tile vision the floor is repainted as a reflected copy of the room:
armchair, window, skyline and books all appear in a surface that has none of
them. With tile vision the same region stays recognisably floor, with plank lines
intact and far less invented content.

That is the failure tile vision was built for. A tile is conditioned on the
prompt for the whole image, so a tile of nothing but floor is told the scene
contains furniture, and at denoise 0.6 it obliges. Showing the tile's own pixels
to the vision tower tells it what is actually there.

## The metric was useless here, again

| | detail energy | wall clock |
|---|---|---|
| tile vision off | 2.027 | 257s |
| tile vision on | 2.051 | 318s |

A 1.2% difference, well inside noise, for a change this obvious by eye. Detail
energy measures sharpness, and an invented reflection is just as sharp as a
floor. This is the second time a number rated these runs wrongly; the first was a
seam-steepness metric preferring a ghosted join. Judge tiling by looking.

## Neither run is clean

Both still show artefacts in the floor, image 02 included. Both were run at
denoise 0.6 with 16 tiles, which is above the 0.35 that
`recommended_maximum_hires_denoise_for_tiling()` reports for a grid this
crowded, so the residual is consistent with that guidance rather than a
contradiction of it. A run at 0.35 has not been made on this subject.
