# Parking Analysis v2.3

For hosting and external apps, see [HOSTING.md](HOSTING.md). The separate hosted entry point exposes authenticated `POST /v1/analyze`; the desktop app keeps its local two-step API. Docker configuration and a private model packager are included; no public deployment is implied by uploading the code to GitHub.

Double-click **START_PARKING.cmd** in the outer folder, or **START_APP.cmd** inside this project. The launcher opens the current app and chooses a free local port if an older server is still running. Check for the **Local app · v2** badge.

Alternatively, run from this folder:

```powershell
python start_app.py
```



## Motorcycle and mixed-vehicle bay coverage (v2.3)

A vehicle-containing bay can be missed even when the vehicle detector sees the motorcycle. Automatic analysis now recovers missing bays from painted divider proposals or bounded gaps between compatible existing bays. Recovery requires an unassigned detected vehicle inside the proposal, little overlap with existing bays, and visible paint along both long sides. A vehicle box alone never creates a bay. Saved manual layouts and explicit corner-only analysis retain their boundaries.

RT-DETR and YOLO11m now provide complementary vehicle evidence in sparse scenes too. Recognized occupancy categories include motorcycles, bicycles, cars, buses and trucks. Existing COCO weights do not have a dedicated van class: vans may be labelled car or truck, while lorries use truck. The parser additionally accepts compatible custom-model labels such as van, motorbike, lorry, pickup and scooter; aliases do not train new visual categories. People are excluded. Duplicated nested vehicle boxes are suppressed before bay assignment. Two motorcycles in the same bay still count as one occupied bay, and a large vehicle genuinely spanning two bays can mark both occupied.

The evidence table shows the detected vehicle categories and identifies recovered painted bays. Final occupancy stays binary: occupied or available. The recovery logic is inference code, not newly trained detector weights. The supplied motorcycle screenshot was reproduced as a 260-pixel-wide source-photo crop: the old pipeline returned two occupied car bays, and the new recovery adds the occupied motorcycle bay. Source images and detailed regression reports stay local under `outputs/vehicle-coverage/`.

This example is not an accuracy certificate for all motorcycles, vans or lorries. Tiny/occluded vehicles, indistinct paint and missing neighbouring bay anchors can still prevent automatic recovery. The existing PKLot benchmark primarily covers cars; independent labelled mixed-vehicle scenes are required to measure motorcycle/van/lorry accuracy. No 99.99999% accuracy claim is made.

## Current binary output and multi-dataset training

Version 2.2 shows only **Occupied** and **Available**. The unknown counter and manual unknown option were removed. Unresolved internal evidence goes through the bay-content classifier, which makes a two-class decision even below its former abstention thresholds. A direct vehicle match still wins. This changes the decision policy; it does not make uncertainty disappear. Intermediate evidence is kept in JSON for auditing.

The active occupancy head was retrained on **22,302 crops** from PKLot, CNRPark and CNR-EXT, including flips and rectangular bay crops to match detector output. Frozen ImageNet MobileNetV3-Small features feed a trained linear head. Linear and nonlinear candidates were compared using validation data; epochs with validation accuracy or false-available regression were excluded. The selected model also passed the final test regression gate. The earlier active model is preserved as `models/occupancy_bays.pklot-only.backup.pt`.

| Evaluation | Before training | Active model |
|---|---:|---:|
| PKLot application, 15 photos / 1,500 labelled bays | 98.53% with binary policy | 99.53% |
| CNRPark held-out camera B, 6,413 crops | 81.10% | 88.68% |
| CNR-EXT held-out cameras/days, 3,000 crops | 95.93% | 98.10% |

PKLot application results: 690/690 occupied bays and 803/810 empty bays correct, seven false-occupied bays, no unknown outputs and zero false-available errors. There are still **198 unmatched bay predictions**, potentially including visible unannotated bays. The photo-bootstrap interval is 98.8%-100%, so the strict 99% gate is **not met**. CNRPark and CNR-EXT results classify supplied crops and do not evaluate automatic bay boundaries; they include 585 and 38 false-available errors respectively. The difficult CNRPark camera remains a material limitation. These results do not establish universal 99% accuracy or real corner-only boundary accuracy.

CNRPark uses camera A time blocks with six frames excluded between training and validation, while camera B on a different capture day is test-only. CNR-EXT additionally separates hashed capture-day groups and cameras: training cameras 1-6, validation camera 7, test cameras 8-9. Exact duplicate CNR images are removed before feature extraction. Manifests are in `datasets/cnrpark/`. PKLot keeps its existing camera/day splits; original parking-detector pretraining overlap is unknown.

Reproduce training and evaluation:

```powershell
python tools/train_multidataset.py --extended --download --output models/occupancy_multidata.new.pt
python tools/validate_application.py --target .99 --output outputs/application-validation/test-current-99.json
```

The training command verifies/downloads the official CNR archives, reuses feature caches, trains candidates and writes reports. It **does not replace active weights automatically**. The GUI's training disclosure has a **Multi-dataset training command** button. `outputs/multidataset-deployment.json` records the deployed checkpoint hash and results. The earlier PKLot-only experiment and rejected candidates are retained for comparison. Source: [authors' CNRPark+EXT dataset](http://cnrpark.it/) and [official release archives](https://github.com/fabiocarrara/deep-parking/releases/tag/archive).

## Analyse a photo

1. Choose a JPEG, PNG or WebP photo and click **Upload & analyse**. No manual marking or training is required for this workflow.
2. Read the red occupied and green available results. Every detected bay receives one of these two labels. The evidence table explains each decision and distinguishes a separate detected vehicle from occupancy inferred by the parking model.
3. Use **Thorough** mode when needed. It combines baseline and higher-resolution bay inference, overlapping vehicle crops and individual checks around predicted empty bays. It can take substantially longer on the CPU.
4. Download the annotated image or JSON. Each result belongs to one immutable photo ID; uploading another photo does not reuse the old boundaries or overwrite its results.

**Edit bay boundaries** is optional. Drag a corner, Shift-click to delete a bay, or click four corners and Finish bay. Save, then analyse while editing is checked to use those boundaries. Manual status corrections are applied to the existing result immediately without rerunning neural inference and are clearly marked manual. Select Automatic to reset a correction.

The supplied-photo selector is a quick test with the images already in this folder. The original photos, existing training exports and public dataset have been preserved.

## Four L-shaped corner brackets (no connecting lines)

Automatic mode now also detects bays defined by **four separated inward-facing white L-shaped brackets**. It fits the observed corners into a quadrilateral, checks arm directions and bay proportions, and rejects combinations that span across intermediate markings. Matching learned rectangles are refined to the observed corner boundary; a broad model box cannot merge two fully marked bays.

To isolate this detector, open **Advanced analysis & model details → Bay boundaries → Four L-shaped corners only**, then analyse. The evidence table tags these as **four-corner boundary**. The selector includes `corner-brackets-demo.png`, a controlled synthetic test with three empty marked bays; it is not a real photograph or a real-world accuracy benchmark.

Corner-only geometry never implies that a bay is empty. Vehicle detections take priority. A new local **MobileNetV3-Small bay-content classifier** provides independent occupancy evidence on rectified corner-bay crops. Unresolved corner bays also receive local parking-model and vehicle-crop checks. Conflicting evidence is resolved to a binary estimate by the bay-content classifier; the evidence is retained for review.

The initial v2.1 occupancy classifier used frozen ImageNet features plus a trained occupancy head, fitted on 2,428 labelled PKLot training bays (70 source photos, with horizontal-flip augmentation). Thresholds were chosen using 516 validation bays from 15 different capture-day photos. The held-out PUCPR camera test contained 1,500 bays: 689 correctly accepted occupied, 788 correctly accepted available, 8 empty bays marked occupied, 15 unknown, and zero occupied bays reported available. The raw top-class accuracy was 99.2%; this is a **crop-classification** result, not an end-to-end corner-detection accuracy claim. Model confidence scores are not guarantees.

Active checkpoint: `models/occupancy_bays.pt`; evaluation details: `models/occupancy_bays.json`. The starting feature weights came from [official Torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), and the current occupancy training includes PKLot and CNRPark. Training reports and activation decisions are recorded in `outputs/multidataset-deployment.json`. No predictions from your imported photos were used as training labels.

Corner geometry tests cover separate bays, 90° and 12° rotation, moderate perspective, dim/blurred/JPEG-compressed paint, a missing corner, and rejection of continuous rectangular outlines. The supplied continuous-line sample photos produced no extra corner-only bays. There is still **no labelled real corner-only photo in this workspace**, so accuracy on your marking style remains unmeasured. Severe perspective, worn/hidden corners, touching T/cross junctions and very small marks may be missed. The learned model remains available in Automatic mode when geometry cannot resolve all four marks.

CLI example:

```powershell
python test_image.py --image "your-corner-marked-photo.jpg" --engine corner_brackets
```

Current multi-dataset classifier training is reproducible with `python tools/train_multidataset.py --extended`; the earlier PKLot-only experiment is `python tools/train_bay_classifier.py`, using the downloaded ImageNet weights and the local PKLot annotation metadata. It writes a candidate and a held-out report; do not replace active weights unless the report passes its deployment checks.

## Models actually used

- **RT-DETR-L** is the primary vehicle detector. This is a transformer-based detector, replacing YOLO26n in the main analysis pipeline.
- The existing parking-specific **YOLO26m** checkpoint discovers bay regions and estimates occupancy. It is separate from generic vehicle detection.
- **YOLO11m** supplements vehicle detection in dense scenes with at least 20 detected bays.
- Painted-line geometry supports white and yellow markings as an explicit advanced fallback, or when no bays can be localized. Geometry supplies boundaries, never empty labels.

YOLO26n remains on disk for historical experiments but is not the default app, CLI, video or notebook detector. The rejected fine-tuning checkpoint also remains inactive. The app displays actual model names; there is no silent fallback to a smaller model when required weights are missing.

Sources: [official RT-DETR support](https://docs.ultralytics.com/models/rtdetr/), [official YOLO11 support](https://docs.ultralytics.com/models/yolo11/), [parking checkpoint](https://huggingface.co/simahanyan/parking-lot-yolo26m). The parking checkpoint is shared for evaluation/feedback; its owner requests contact before commercial redistribution or deployment. Other model licensing also applies; this is a local evaluation prototype.

## What changed and what remains uncertain

The previous app had duplicated inference paths, model controls that did not affect the main parking model, an obsolete YOLO26n title, a missing default sample path, shared result filenames and layouts detached from image identity. The rebuilt app uses one inference pipeline across browser, image CLI, video and notebook, with image-bound state and distinct result downloads.

A vehicle is assigned primarily to its best-matching bay. Vehicle overlap, clipped boundaries and conflicting evidence are checked internally. The final binary resolver uses the bay-content classifier to choose occupied or available for unresolved bays. Missing car detections alone never establish availability. If the occupancy model is missing, unresolved bays are conservatively labelled occupied and the reason says so. Rectangular model regions can still overlap or miss the actual painted boundaries, especially in angled scenes.

The detector is not an LLM. It works best on real photographs resembling its training views. Drawings, severe occlusion, faded markings and extreme viewpoints can still fail. The app always outputs a binary estimate; forcing a decision can produce incorrect available or occupied labels. Confidence scores are not calibrated guarantees. No overall accuracy claim is made from unlabelled sample predictions.

Observed comparisons on supplied photos are in `outputs/model-upgrade/`. RT-DETR detected 5 vehicles in the larger overhead image and 8 in the angled two-row image where the tested generic YOLO models detected none. The parking model supplies additional occupied-bay evidence. The browser flow was checked with real model inference, image switching, photo upload, evidence display and reversible manual corrections. These smoke checks are not a labelled accuracy benchmark.

## Installation and model recovery

Dependencies and the required checkpoints are already installed in this workspace. On another machine:

```powershell
python -m pip install -r requirements.txt
python setup_models.py
python start_app.py
```

`python setup_models.py --check` verifies checkpoint SHA256 hashes. Inference is local and uses the CPU. The first analysis takes longer while models load. Use `python start_app.py --no-browser --port 8766` for an explicit server port without opening a browser.

## Shared CLI and notebook

```powershell
python test_image.py --image "samples/parking photos/parking.jpg"
python test_image.py --image "samples/parking photos/parking4.jpg" --quality accurate
python test_video.py --source samples/test.mp4 --interval 1
```

The image CLI automatically discovers bays. Supply `--layout path.json` only for a reviewed layout belonging to that exact view. The notebook `notebooks/Parking_Lite_Test.ipynb` uses the same pipeline. The older OpenCV boundary mapper remains available as `tools/map_spaces.py`.

## Training tools

Training is separate from ordinary analysis and lives under the **Prepare and review training photos** disclosure. Existing photo queues and reviewed exports are preserved under `datasets/preparation/`.

```powershell
python tools/prepare_training.py --input "C:\path\to\photos"
python tools/build_training_dataset.py --status
python tools/build_training_dataset.py
```

You can fill AI suggestions for review, correct them, mark the photo fully reviewed and export. Predictions alone are not training ground truth. The dataset builder uses only complete human-reviewed exports and keeps each source photo in one split. Related capture sessions also need separation for a real accuracy study.

The earlier 100-photo PKLot experiment remains under `datasets/public_pklot_100/`, with 70 training / 15 validation / 15 test photos and 4,444 annotated bays. Five fine-tuning epochs completed, but the candidate performed worse and was not activated. Historical metrics and provenance are in `outputs/public-finetune/comparison.json`; they describe the earlier pipeline, not a certification of v2 accuracy. PKLot may overlap the original checkpoint's pretraining data.

A reproducible public-data experiment is available with `python tools/train_public_parking.py --count 100 --epochs 5 --activate-if-better`. It retains the original checkpoint when the candidate fails its evaluation gate. Improving performance on the actual deployment views requires independently reviewed images from those views.

## Files and validation

`data/photos/<photo-id>/` stores each uploaded image, its optional layout and immutable results. `models/` stores checkpoint files. The old `outputs/` and `samples/` artifacts have been preserved. This is a single-user, localhost-only app without authentication; it is not a multi-camera production service.

Run `python -m pytest -q`. Regression checks cover stale-image rejection, immutable results, bay/vehicle assignment, false-green overlap protection, Thorough-mode crop checks, corrections without reinference, exports and training split integrity.

### Reproduce the 80% application acceptance check

```powershell
python tools/validate_application.py --output outputs/application-validation/test-current.json
```

This runs the actual default application pipeline, including vehicle assignment and uncertainty guards, on the 15 labelled PKLot test photos. A ground-truth bay succeeds only if a unique predicted bay overlaps its bounding box with IoU >= 0.5 and has the correct occupancy. Missed bays and unknown statuses count as failures. Acceptance requires at least 80% joint success, at least 80% recall for each occupancy class, and a photo-cluster bootstrap 95% interval lower bound of at least 80%. The JSON records model hashes, per-photo counts, false-available errors and unmatched predictions. Unmatched predictions are reported separately because the dataset can leave visible bays unannotated.

This is a dataset-specific acceptance check, not a promise for arbitrary uploaded photos. The existing test subset was used in earlier experiments, original parking-checkpoint pretraining overlap is unknown, and its photos do not validate real L-shaped-corner-only bays. The bootstrap interval measures variation across the supplied photos, not across unseen camera types. Keep new deployment lots and capture sessions separate from training and validation before claiming deployment accuracy.

### Historical v2.1 neighbouring-car overlap improvement

The default engine now uses the bay classifier to corroborate some empty bays previously left unknown by the vehicle-box overlap guard. This requires an existing positive empty-bay prediction, classifier empty score >= 0.995, a vehicle assigned to another bay with its centre outside this bay, and overlap below 35% of the smaller area. Larger overlaps, conflicting occupied predictions and weak classifier evidence retain the original safeguards. This score is a model output, not a calibrated probability of real-world correctness. No model weights or confidence thresholds were lowered.

Validation joint accuracy increased from 96.3% to 98.1%; test joint accuracy increased from 95.3% to 97.2%. Test occupied recall stayed 100%, empty-bay recall improved from 91.2% to 94.8%, and no occupied bay was marked available. The test photo-bootstrap interval is 94.7–98.9%. There remain 198 unmatched predictions, which can include unannotated visible bays. See `outputs/application-validation/overlap-upgrade.json` for activation evidence and the before/after reports alongside it.

The stricter 95% gate is **not met**: empty-bay recall and the bootstrap lower bound remain below 95%. Reproduce that check with `python tools/validate_application.py --target .95 --output outputs/application-validation/test-current-95.json`. Use `--no-verify-boundary-overlap` to compare the previous behaviour. The browser and CLI use the improved behaviour by default after restarting the application.
