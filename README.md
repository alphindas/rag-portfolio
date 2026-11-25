# Forklift load volume from one video

**Problem.** In a public Linde video (https://www.youtube.com/watch?v=RCqa8Kojlw8, pass-by at about 20-22 s) a forklift drives past with a
load of stacked crates on pallets. Estimate the load volume in m³ from the video alone. This is a prototype.

**Run it:** `pip install -r requirements.txt`, then open `notebook.ipynb` and run all cells (or `python src/measure.py`, `python src/uncertainty.py`, `python src/make_results.py`; tests: `python -m pytest tests -q`). Final numbers: `outputs/results.json`.

## Result (short)
* **Most likely scenario D (3 EUR pallets in a row, 2.4 x 1.2 m, 4 layers): about 3.6 m³ of load (80 % range 3.4-3.7 m³ inside D).** Load plus pallets about 4.0 m³.
* **Across all scenarios that survive: about 1.8-3.7 m³**, depending on how many pallets are on the forks (C about 1.8, B about 2.4, D about 3.6; the 1.8 is C's median). A (1.14) is excluded.
* **What decides it:** the number of pallets and the fork length (D needs a third pallet beyond 1200/1800 mm forks), plus the crate type. None of these is verified from the video alone.
* The ranges inside a scenario are small (about 5 %); the choice of scenario is the big uncertainty (see "Two kinds of uncertainty").

> **Status: no single final number.** Scenarios B, C and D are carried side by side, none is "main". The reason is two unverified facts:
> **the crate type and the pallet layout in this video are NOT verified** (see Assumptions 1 and 2). Case A (one EUR pallet) is excluded.

## Method in plain steps
1. Download the video (`yt-dlp`, not stored in the repo) and pick sharp frames of the pass (`src/extract_frames.py`).
2. Click corner points by hand on frames #1086, #1092, #1098 (`src/annotate.py`). #1092 was clicked twice to measure click error (mean 1.7 px, worst 9.8 px).
3. Scale (mm per pixel) from the truck datasheet: mast height 2888 mm over the pixels from mast top to the floor under the front wheel. This is the **height ruler**.
   The pallet height (144 mm on only about 30 px) is a **weak cross-check**: 1 px of error is about 3 %, the top edge is hard to see, and click error alone spreads the pallet-height
   scale by about 15 % against about 1 % for the truck scale. The two scales disagree by 22-27 % (pallet scale larger), which is the main open problem.
4. Load height = pixels x vertical scale. Length of the long face = pixels x scale **plus perspective**: the nearer end looks taller, the height ratio (1.15-1.17) tells how far the face is turned away
   (about 22-25°). The truck gives a focal length of at most about 1270-1390 px (its clicked length against the datasheet length). `src/measure.py`
5. Crate checks: count layers and crates, compare with real crate sizes (`src/crates.py`).
6. Monte Carlo, 10,000 runs, inputs: click error, scale bias, fill factor, pallet tolerance, focal length (`src/uncertainty.py`).

## Scenarios (width and crate type are not certain; none is chosen)
Load-only volume, 80 % range from the Monte Carlo (`outputs/uncertainty.json`). Frames #1086/#1092/#1098 (clicked), pooled. Heights now use the **left-end** height with the truck scale
(1.25 m = 4 layers x 312 mm); an earlier version used the mean height and understated it by about 8 %.

| | A: 1 EUR pallet | B: 2 pallets in a row (8 crates x 300 mm) | C: length from clicks | D: 3 pallets, 800 mm side along the forks |
|---|---|---|---|---|
| Length x width | 1.2 x 0.8 m | 2.4 x 0.8 m | about 1.8 x 0.8 m | 2.4 x 1.2 m |
| Height | 1.25 m (image) | 4 x 0.29-0.36 m | 1.25 m (image) | 4 x 0.310-0.333 m (**assumption**, see below: catalogue range of a 12 x 0.75 L crate 400x266, not measured on this crate) |
| Load only | 1.14 (1.04-1.24) | 2.37 (2.15-2.59) | 1.75 (1.47-2.06) | **3.56 (3.38-3.74)**; crates x crate volume (108 crates): 3.70 (3.59-3.80) |
| Load + pallets | 1.27 | 2.65 | 1.96 | 3.97 |
| Width 1.2 m instead of 0.8 m | 1.70 | 3.55 | 2.62 | (already 1.2 m) |

### New clicks: frames #1062 and #1068 (`src/new_frames.py`, `outputs/new_frames.json`)
The right end of the load is cut off by the frame edge in both, so only the left-end height and a length lower bound can be measured. Frame #999 was skipped for the same reason.
* **Height (truck scale, left end):** #1062 1.29 m, #1068 1.28 m (per layer 323 / 320 mm), against 1.24-1.25 m in #1086/#1092/#1098. All 5 frames: 1.24-1.29 m, mean 1.26 m (spread 4 %). The change is about +3 %, well under 10 %, so the Monte Carlo
  numbers above are **not changed**; the per-layer height of 320-323 mm lies inside D's assumed 310-333 mm.
* **Length:** in #1068 the top edge is at least 597 px long (clicked right end at x = 1903, at the frame edge), which is at least **1.96 m** at the left-end scale (±15 % for the unknown depth of the right end). So the load is **longer than about 1.9 m**.
  This is a lower bound only: it leaves C (about 1.8 m, 1.47-2.06) very tight and does not separate B from D (both 2.4 m). D's length (3 x 0.8 m) comes from the counts and is unchanged.
* The near-frontal frame did not give a ratio (right-end clicks missing), so the turn angle is still from #1086-#1098.

### Two kinds of uncertainty (do not mix them)
**(a) Inside one scenario.** The Monte Carlo ranges above (for D: 3.38-3.74 m³, crates x volume 3.59-3.80) only cover click error, scale bias, fill factor, pallet tolerance, gap and focal length.
They assume that the scenario itself is right. They are the *small* uncertainty.

**(b) Which scenario is right.** This is the *large* one and the Monte Carlo does not cover it: the load-only medians run from 1.75 (C) over 2.37 (B) to 3.56 m³ (D), a factor of about 2 between the ends,
and A (1.14) is excluded. It depends on facts that are not verified: the crate type, how many pallets are on the forks, and the load depth (0.8 or 1.2 m). Today the evidence favours D, but D still has an open problem
(the fork tips, see the consistency table). "They carry several pallets together" is **likely** (blue-load seam, 5-tine carriage at 52 s, see Assumption 14) but **unverified**; Carolinen could not be asked.
So D's range must not be quoted as the whole uncertainty. The honest statement today is "about 1.8-3.7 m³ depending on the scenario, D most likely, and within D 3.4-3.7 m³".

### Consistency table (what each scenario has to explain)
Facts read from the frames: **9 crate columns on every row** (frame #999, where the left end is visible), **4 layers**, **3 white labels, one per 3 columns** (spacing 3.1, 3.3, 3.0 columns in #999; 3.2 in #1008; 3.0 in #1092),
**about 9 pallet feet** in 3 groups of 3 (8 clearly seen in #999 plus one at the frame edge; two pairs of feet side by side where two pallets would meet).

| Test | A | B | C | D |
|---|---|---|---|---|
| 9 columns, 3 labels, 9 feet | no | no (expects 8 columns, 2 labels, 6 feet) | not tested | **yes** |
| Length vs measured (#999: 2.45-2.72 m, proposed points; #1092: 2.05-2.20 pallet scale, 1.79-1.92 truck scale) | no (1.2 m) | yes for #999 and pallet scale, 25 % long for #1092 truck scale | yes for #1092 truck scale, 30 % short for #999 | same as B |
| Height 1.25-1.5 m measured (4 layers = 310-375 mm per layer) | ok | ok | ok | **yes** (1.24-1.33 m) |
| Crate cell width/height measured 0.81-0.91 (apparent, from window pitch) | - | 300 x 290 needs a 24-37 deg turn (turn estimate 12-27 deg): borderline | borderline | 266 x 310-333 predicts 0.82-0.88: **fits** (needs a nearly frontal face) |
| Fork tips (datasheet 1200 or 1800 mm) | fits | 2.4 m deep: overhangs 1.2 / 0.6 m | 1.8 m fits 1800 mm forks | same as B: pallet 3 hangs 1.2 m (1200 mm forks) or 0.6 m (1800 mm forks) beyond the tips |
| Verified crate for the count | - | 300 mm side: 20 x 0.5 L only | no crate with a 200 mm side along 9 columns in 1.8 m (9 x 0.2) | 12 x 0.75 L crate 400 x 266 exists (Delbrouck), 9 per layer and 4 layers as in the Wasserhelden pallet data |

**What survives.** A is excluded (counts and length). B as defined does not survive the counts (it needs 8 columns, not 9). C survives only on one frame's truck scale and conflicts with the counts, with frame #999, and with
a plain 266/300 crate. D survives the counts, the length (on 3 of 4 estimates), the height and the crate type, but it has to explain why the third pallet hangs over the fork tips, and it differs by 20-25 % from the #1092 truck-scale length.
The datasheet lists no fork longer than 1800 mm for the E80 but mentions "hook-on and integrated attachments" without sizes, so a longer reach is possible but unverified. No scenario is selected here.

* **Same load, two views.** In #999 the right end is nearer (edge height ratio 0.80); in #1092 the left end is nearer (1.17). The truck swings during the pass. With the pallet-height scale the two frames give 2.45-2.56 m and 2.05-2.20 m
  (12 % apart); with the truck scale 2.60-2.72 m and 1.79-1.92 m (about 40 % apart). So the truck scale at #1092 is the least reliable number. #999 values come from points **read by eye by Claude, not clicked by you**.
* Case A is excluded: the pallet long edge is about 14 times its height in the image (a 1200 x 144 mm face would be 8.3).

## Where the crate type came from (and did not)
* Stickers on the crates were cropped at full resolution and upscaled 6 times (`outputs/stickers_sheet.png`): they show paper labels with text lines and two barcodes. **Nothing is readable** (no brand, pack size or article number).
* The video's title card says "LINDE E80 IN DER GETRÄNKEINDUSTRIE, Carolinenbrunnen, Bielefeld"; a caption names Mineralquellen Wüllner GmbH & Co. KG. So the plant is known, the crate is not.
* No Carolinen datasheet with crate size was found. REWE product pages could not be read by script.
* Real crates that exist (manufacturer catalogue Delbrouck, read directly): 20 x 0.5 L 400x300x290-302; 12 x 0.75 L 400x266x310-333 and 400x300x291-360; 12 x 1 L 400x300x310-330 and 400x266x314; 12 x 0.7 L 375x266x305.
* A returnable GDB 0.75 L crate of another brand: 9 crates per layer, 4 layers, EUR pallet (Wasserhelden "Artikelpass"). Layers match our 4; the crate size is not given there.
* Blue crate pallet in the background (about 93 s, light blue, corner view, partly hidden by the truck): 5 layers, at least 6-7 columns on its long face. **Not conclusive** for a crate size.

## Assumptions (every one of these is a guess or an unverified fact)
1. **Crate type is NOT verified for this video.** Heights of 0.29-0.36 m are the range of real 20 x 0.5 L and 12-bottle crates (sources below), not a match for the video's crate.
2. The pallets are EUR pallets (1200 x 800 x 144 mm). No EPAL mark or block count could be confirmed. +-3 mm per pallet and a 0-30 mm gap between two pallets are assumed.
3. Load width (depth into the picture) is 0.8 m (alt. 1.2 m). Not visible in this view.
4. Forklift model "Linde E80" comes from the video's own title card. Datasheet: Linde E60/E80, series 1279 (4.2 mast height lowered 2888 mm; 4.20 length to fork face 3503 mm; 4.22 forks 1200 mm for the
   600 mm load-centre version, 1800 mm for the 900 mm version; 4.24 carriage width 2180 mm). Which variant is in the video is **not known**. Fork spacing is not in the datasheet.
5. Truck scale error from different depths (mast vs. wheel): +-10 % (assumed). The pallet-height scale disagrees by 22-27 %, outside this band, so that gap is carried as the two scenarios.
6. Camera: ideal pinhole, image centre as principal point, no lens distortion; focal length 800-1400 px (upper limit from the truck geometry, lower limit a wide-angle guess).
7. The long face is a flat rectangle with equal vertical edges (the height ratio left/right comes from perspective only).
8. Fill factor 0.90-1.00 (share of the bounding box that is filled). Click error: 2-5 px for sharp points, 10 px for blurry points.
9. Crate count: 4 layers (counted by two people, re-checked on a close crop), 8 crates along the lower rows (9 in the top row, left end hidden by the fork carriage).
10. Frame #1098: the "truck rear" click is a different feature than in the other frames; it is not used for volumes.
11. Frames #1086-#1098 are only 0.1-0.24 s apart; their errors are correlated. Frame #999 (19.98 s) shows the whole face from the other side of the turn; its corner points were **read by eye by Claude and are not verified** (`src/early_frames.py`).
12. Extra frames used (not stored in the repo, see timestamps): green-crate close-up 16.0-17.5 s; blue-crate loads 115.4 s, 117.1 s, 138.5-139.6 s, 140-142 s; early pass 19.7-20.3 s. In the close-up the bottle labels read "Carolinen naturelle".
14. **Empty forks found (video 51-52 s, frames 02575-02625 in `extra_pics/`):** a Linde truck of the same family with a very wide carriage and **5 fork tines** (so a multi-pallet fork is in use at this plant).
   It is an oblique view and probably a different truck, so it does not prove the pass-by truck has the same forks. Fork length cannot be read from this shot: the tines point partly towards the camera, and with the unknown turn
   angle the image gives only the relation length = 1.4 m / tan(angle between the forks and the camera direction), from about 290 px of sideways tine extent against about 445 px of carriage width (2180 mm datasheet). No second view with a visible fork tip at the far end of the pass-by load was found.
15. **Crate height 310-333 mm in D comes from a catalogue range** (Delbrouck, 12 x 0.75 L, 400 x 266 x 310-333, URL in Sources). It is the height range of products that exist, **not verified for the crate in this video**. The measured load height (1.24-1.33 m, truck scale) is consistent with 4 layers of that range, but that is a consistency check, not a source.
13. Blue-crate loads (Carolinen crates, 5 layers) show a seam in the middle of the pallet and 3 crates per pallet along a face, with 6 crates across in the front view. This suggests the plant carries several pallets together, but it does not fix the crate size.

## Sources
* Linde E60/E80 datasheet (series 1279): https://www.linde-mh.com/media/Datasheets/EN_ds_e60_e80_1279_en_b_0224_view.pdf
* IPL Schoeller crates 3150001 (400x300x290) and 3283001 (400x300x300): https://www.iplschoeller.com/products/3150001-beverage-crate-20x50cl , https://www.iplschoeller.com/ch/produkte/3283001-beverage-crate-20x50cl
* Delbrouck crate catalogue: https://www.delbrouck.de/en/getraenkekaesten/
* Paderborner datasheets (20 x 0.5 L crate 40x30x30 cm, 8 per layer, 5 layers, 161 cm with pallet): https://paderborner-brauerei.de/wp-content/uploads/2025/08/2025_08_26_PB-Export_Datenblatt_20x05l_web.pdf , https://paderborner-brauerei.de/wp-content/uploads/2024/02/PB_Limo_Cola-Orange_DB_Fl_05lx20.pdf
* Wasserhelden Artikelpass (0.75 L GDB crate, 9 per layer, 4 layers): https://produktinfo.blauer-engel.de/uploads/pdf_uploads/Artikelpass%20Wasserhelden_ohne%20Preise.pdf
* Video: https://www.youtube.com/watch?v=RCqa8Kojlw8 (frames at 21.72 s, 21.84 s, 21.96 s)

## Where this method would fail
* **Load is not a box** (gaps, overhang, shrink-wrap, a half-empty top layer): the box volume is then too high; the fill factor 0.90-1.00 is a guess.
* **Hidden or non-EUR pallet**, or a different pallet height than 144 mm: the pallet-height scale breaks (it is already the weak scale, about 30 px).
* **Depth (load width) is a guess** (0.8 or 1.2 m, not visible from the side). This alone changes the volume by 50 %.
* **Wrong number of pallets / crate type** (the main open point here): the volume scales with length.
* **Steep or unknown camera angle, lens distortion:** one-camera perspective is simplified (pinhole, no distortion); a wide-angle lens bends the edges. Focal length is only bounded (800-1400 px).
* **Truck scale at different depths** (mast vs. load): up to about 10 % assumed; it was 40 % off for the length in one frame.
* **Motion blur and cut-off frames:** click error 2-10 px; the right end of the load was out of frame in several frames.
* **Next steps:** a second camera or a depth sensor (gives width and true length directly), a frame with the pallets and fork tips visible, a known reference object in the load plane, a crate datasheet from the plant.

## Still open
Frames #999 and #1014 were not clicked (the right end is cut off), so the #999 points in `src/early_frames.py` stay Claude's eyeballed, **unverified** readings. Crate type, number of pallets and fork length of the pass-by truck are not verified (the plant could not be asked).
