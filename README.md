# SPS — Smart Piping System

SPS is an early-stage research prototype for exploring water-system sensor data
and classifying normal versus anomalous event samples. The current code is an
offline analysis/training workflow; it is **not** a deployed residential leak
detector.

The original hardware concept mentions Arduino and an HW-038 water-level sensor.
The HW-038 is not a flow meter, and this repository does not currently connect
to Arduino hardware, consume a live sensor stream, or localize a problem to a
room or floor. The available WDSEventDB measurements come from a laboratory
water-distribution testbed, not residential SPS installations.

## Current status and results

The supervised experiment uses the `Labels` column as its target and sensor
readings as model inputs. The project-provided paper excerpt describes Labels
0 as normal samples and Labels 1 as anomalous event samples; the paper page was
not independently accessible during the project audit. In any case, an
anomalous event is not necessarily a leak in every scenario. The following
metrics are event/anomaly results on WDSEventDB, **not leak-detection accuracy**.

- Random row-split Random Forest baseline: F1 **0.9963**. Rows from the same
  workbooks can be present in both splits, so this is not independent
  workbook/event validation and may be optimistic.
- Leave-one-Leakage-workbook-out evaluation: F1 **0.0019**, recall **0.0009**,
  balanced accuracy **0.5005** at the default decision threshold.
- Training-only feature selection changed grouped F1 to **0.0037** (4 versus 2
  detected anomalous samples out of 2,138); this did not solve generalization.
- The threshold-calibration experiment selected a separate threshold per fold
  using a different Leakage workbook for validation. Its pooled held-out
  precision was **0.7271**, recall **0.9860**, F1 **0.8370**, and balanced
  accuracy **0.7922**, with **791 false positives among 1,970 normal samples**.
  The selected thresholds varied by fold (0.01, about 0.00333, and 0.14).
  This suggests the default threshold contributed to missed positives in this
  specific protocol, but the high false-positive count and changing thresholds
  are important limitations.
- Threshold-calibration folds train on Clean plus one Leakage workbook, while
  the grouped evaluation trains on Clean plus two Leakage workbooks. The
  protocols therefore differ in training data; these results do not isolate
  threshold as the only difference from the earlier grouped evaluation, and do
  not establish that calibration fixes cross-workbook distribution shift.

Rows have no per-row timestamps in the available workbooks, so temporal
independence and chronological generalization are not established. The small
number of workbooks also limits conclusions. Do not describe the current model
as ready for real-world leak detection or use these results as proof of
residential performance.

## Repository layout

```text
data/
  raw/                         Local source data (not tracked by default)
  processed/WDSEventDB/        Extracted workbooks (not tracked by default)
models/                        Serialized model (not tracked by default)
reports/                       Audit and experiment reports; retained in Git
scripts/                       Audit, preparation, training, and evaluation code
requirements.txt               Direct third-party Python dependencies
```

### Scripts and reports

- `scripts/audit_data.py` — read-only audit of `data/raw/water_network.csv`;
  uses the Python standard library.
- `scripts/prepare_data.py` — validates the nine WDSEventDB workbooks and
  prints schema, labels, missing/duplicate rows, and timestamp observations;
  uses the Python standard library.
- `scripts/explore_leakage_data.py` — compares Clean and Leakage sensor
  measurements and writes `reports/leakage_exploration.txt`.
- `scripts/train_baseline.py` — trains the current Random Forest on sensor
  readings with `Labels` as target, using a stratified random row split. It
  writes `models/sps_event_baseline.joblib` and
  `reports/sps_supervised_baseline_results.txt`.
- `scripts/evaluate_grouped.py` — leave-one-Leakage-workbook-out evaluation;
  writes `reports/sps_grouped_evaluation.txt`.
- `scripts/diagnose_group_shift.py` — descriptive workbook/label distribution
  comparisons; writes `reports/sps_group_shift_diagnosis.txt`.
- `scripts/evaluate_feature_selection.py` — compares all features with
  training-fold-only selected features; writes
  `reports/sps_feature_selection_results.txt`.
- `scripts/evaluate_threshold_calibration.py` — grouped validation threshold
  calibration experiment; writes `reports/sps_threshold_calibration.txt`.
- `scripts/sps_research_diagnosis.py` — compiles the research/protocol audit
  into `reports/sps_research_diagnosis.txt`.
- `reports/baseline_experiment.txt` and `reports/sps_baseline_results.txt`
  document earlier feasibility and unsupervised exploratory work. The remaining
  reports preserve the corresponding analyses and results; they are not
  duplicate files.

Reports are kept because they document distinct stages and protocols. Some
scripts write to fixed report paths, and `train_baseline.py` also overwrites
the saved model. Back up these outputs before rerunning scripts that write
them. `evaluate_grouped.py` and `evaluate_threshold_calibration.py` create
their report with exclusive file creation and will stop if that report already
exists.

## Requirements

- Python **3.10 or later** (the source uses modern union type syntax).
- Direct dependencies in `requirements.txt`: NumPy, scikit-learn, and joblib
  for inference/analysis, plus Streamlit and Plotly for the interactive
  dashboard and 3D view. The project environment reported NumPy 2.0.2,
  scikit-learn 1.5.2, and joblib 1.4.2; Streamlit and Plotly use compatible
  version ranges in the requirements file.
- XLSX parsing in the data scripts uses Python’s built-in ZIP/XML libraries;
  no Excel application or extra workbook package is needed. The dashboard
  does not add pandas or openpyxl as dependencies.

## Data sources, storage, and licensing

The datasets and generated model remain on the developer’s machine and are
excluded by `.gitignore` by default. This is a publishing precaution, not a
deletion.

### WDSEventDB

- Official repository: <https://github.com/Moazeni-s-Concise-Lab/WDSEventDB>
- Zenodo record: <https://zenodo.org/records/17547955>
- The Zenodo record describes a sensor-instrumented water-distribution
  testbed and declares **CC BY 4.0** for the dataset. The GitHub repository’s
  `LICENSE` says **CC0 1.0 Universal**. These declarations differ; the precise
  scope/relationship has not been resolved. Confirm with the authors and retain
  required attribution before redistributing any archive or extracted data.
- The archive and extracted XLSX files are not tracked by default. Do not
  assume the repository license alone settles the dataset’s reuse terms.

### Water-network CSV

- Zenodo record: <https://zenodo.org/records/15096167>
- DOI: <https://doi.org/10.5281/zenodo.15096167>
- The record declares **CC BY 4.0** and describes hourly, anonymized urban
  water-network data with a `fault_d7` label indicating proximity within
  ±7 days of registered leak events. This is not a same-hour leak label.
- `data/raw/water_network.csv` is also excluded by default. Follow the
  attribution and license terms on the record if reusing or redistributing it.

### Model artifact

`models/sps_event_baseline.joblib` is a locally generated serialized model and
is ignored by default. It is small enough for ordinary GitHub file-size limits,
but it is a binary pickle-based artifact: only load artifacts from a trusted
source. Confirm dataset-derived artifact redistribution rights and the
scikit-learn/runtime compatibility before publishing it. The model is not
needed to reproduce training from source and data.

## Clean setup

Clone the GitHub project into a new directory, then from PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Obtain WDSEventDB from its official Zenodo record after reviewing its terms.
Save the archive as `data\raw\WDSEventDB.zip`, then extract its contents:

```powershell
New-Item -ItemType Directory -Force data\raw, data\processed | Out-Null
Expand-Archive -LiteralPath data\raw\WDSEventDB.zip -DestinationPath data\processed
```

The archive contains the `WDSEventDB` directory and macOS metadata. The
preparation scripts inspect `.xlsx` files and ignore non-XLSX entries. The
expected files are under `data\processed\WDSEventDB\`.

The optional water-network CSV can be downloaded from the file link on its
official Zenodo record to `data\raw\water_network.csv`; check its attribution
terms before redistribution.

## Running existing scripts

From the project root:

```powershell
python scripts\prepare_data.py
python scripts\audit_data.py
python scripts\explore_leakage_data.py
python scripts\train_baseline.py
python scripts\evaluate_grouped.py
python scripts\diagnose_group_shift.py
python scripts\evaluate_feature_selection.py
python scripts\evaluate_threshold_calibration.py
python scripts\sps_research_diagnosis.py
```

`audit_data.py` expects the optional Water Network CSV at its default path; pass
an alternate path as its first argument if needed. The other WDSEventDB scripts
expect the extracted workbooks in the paths above.

These are existing scripts, not a single end-to-end command. Some write fixed
report paths, the training script overwrites its model/report, and some
evaluators require their output report not to exist. Preserve or back up
existing results before running them. The row-split training script is included
for reproducibility only; its score is not a defensible estimate of
cross-workbook or real-world performance.

## Simulated dataset stream

`sps/simulated_stream.py` replays rows from the four Clean/Leakage WDSEventDB
workbooks through the existing saved-model inference pipeline. This is a
dataset simulation, **not** an Arduino connection, live sensor feed, or
residential leak detector. Each emitted reading keeps its relative source
workbook, zero-based data-row index, original `Labels` value as metadata, and
the model's normal/anomalous-event prediction. `Labels` is never sent as a
model feature. The workbooks have no original per-row timestamps, so emitted
records include a separately named UTC simulation emission time.

Run a short, progressive replay from the project root:

```powershell
python -c "from sps.simulated_stream import SimulatedSensorStream; s = SimulatedSensorStream.from_wdseventdb(batch_size=50, interval_seconds=0.1); s.start(lambda b: print(f'Batch {b.batch_index}: {len(b.readings)} simulated readings')); print(s.wait())"
```

`start(on_batch)` invokes the callback on a worker thread. Call `stop()` to
interrupt the replay, or `wait()` to block until it completes and to surface
worker errors. A stream instance is single-use; create a new instance to replay
from the beginning. The saved model predicts normal versus anomalous event
samples; class 1 is not a confirmed water leak.

### Backend inference check

From the project root, run a single in-memory prediction through the saved
backend without launching Streamlit or changing the model:

```powershell
python -c "from sps.inference import SPSInference; engine = SPSInference(); sample = {name: 0.25 for name in engine.feature_names}; print(engine.predict(sample))"
```

Run the backend and project tests with:

```powershell
python -m unittest discover -s tests -v
```

The inference and replay tests load the saved model, so they need the
scikit-learn version it was saved with (1.5.2). A newer scikit-learn can load
the file but fails when predicting; that is an environment mismatch, not a
model change. The pipeline-view and demo-scenario tests need Plotly installed.

`SPSInference` requires exactly the saved model's 15 sensor features, keyed by
name; mapping order does not affect the model's saved feature order. It rejects
missing/extra features, non-numeric values, booleans, NaN, and infinity. Its
output is a binary prediction for the source `Labels` target (normal sample
versus anomalous event sample). The current dataset and labels do not establish
confirmed leak detection or real-world pipe-segment localization. Synthetic
localization is described below; it does not validate field performance. There
is no real-time sensor ingestion. A successful smoke check verifies the
software path and artifact compatibility only, not detection performance.

### Synthetic segment-localization backend

`sps/simulated_localization.py` provides a separate, deterministic sensor
stream over a small branched pipe network, with timestamped frames emitted by
`iter_scenario_frames()`. Scenario truth (the simulated
leaking segment and onset step) is stored separately from sensor readings.
`estimate_affected_segment()` accepts readings and a normal reference, not
scenario truth. It scores each segment using the positive standardized residual
on its linked flow sensor and the pressure drop at that segment's endpoint
sensors. The network must provide one flow sensor per segment and pressure
sensors at segment endpoints. It returns an estimated segment, diagnostic
score and runner-up margin, status, and sensor evidence; scores are not
calibrated probabilities. Weak signals return `no_leak_pattern`, while missing
local flow evidence returns `insufficient_evidence` and competing segment
scores return `ambiguous`.

Example, run from the project root:

```powershell
python -c "from sps.simulated_localization import SimulatedScenario, iter_scenario_frames, estimate_affected_segment; scenario = SimulatedScenario('leak-p03', 'P-03', onset_step=8, seed=7); frames = iter_scenario_frames(scenario); [(print(frame.time_seconds, frame.ground_truth.leak_segment_id, estimate_affected_segment(frame.sensor_readings, frame.normal_reference).estimated_segment_id)) for frame in frames]"
```

In this example, the middle printed value is simulation ground truth and is
not passed to the estimator; the last value is the estimated segment.

This is a transparent topology-linked heuristic, not a hydraulic solver. The
simulator still uses explicit response signatures, while the independent
evaluation constructs responses from graph distances and varies sensor gains
and measurement noise. That evaluation is still synthetic and relies on
assumed pressure/flow behavior; it does not establish real-world leak-location
accuracy. The dashboard's demonstration view consumes this synthetic stream.
The existing recorded-data replay and saved model remain separate and
unchanged.

Run its focused tests with:

```powershell
python -m unittest tests.test_simulated_localization -v
```

### Water-level monitoring (simulated)

`sps/water_level.py` adds a threshold check on one water-level signal
(sensor ID `WATER_LEVEL_RESERVOIR`, values in unspecified project units).
The original hardware concept names an HW-038 water-level sensor, but **no
physical water-level sensor is connected**: every water-level value in the
project is generated by `simulate_water_level_reading()` alongside the
synthetic localization scenario, and its response to a simulated leak is an
assumption, not a measured behaviour.

The rule, with the threshold configurable in `WaterLevelConfig`
(default `200`):

| Reading | Condition | Meaning |
|---|---|---|
| above 200 | `normal` | Normal |
| below 200 | `potential_leak_indication` | Potential leak indication (not a confirmed leak) |
| exactly 200 | `boundary_review` | Boundary condition: does not open an alert and does not count toward clearing one |
| missing / non-numeric / NaN / outside configured min–max | `unavailable` / `invalid` / `out_of_range` | No alert and no change to an open event |

`WaterLevelMonitor` is stateful so one continuous low-level event produces
exactly one alert: `alert_triggered` is true only on the reading that opens
the event, and the event closes after `clear_after_normal_readings`
consecutive readings above the threshold (default 3), so readings hovering
around 200 do not raise repeated alerts. `summarize_water_level_events()`
collapses a run of assessments into one record per event (start step, end
step or still active, lowest reading, readings below threshold); the
dashboard alert list is built from these records.

Separation from the existing analysis:

- Water level is **not** a feature of the saved model. `SPSInference`, its
  15-feature schema, `models/sps_event_baseline.joblib`, the datasets and
  the WDSEventDB replay are unchanged.
- Water level is **not** part of the localization sensor vector:
  `estimate_affected_segment()` still uses only the 11 pipe-network sensors
  and rejects an extra water-level key.
- The simulated water-level noise uses its own random stream
  (`water_level_rng()`), so a scenario's pipe-sensor readings are identical
  with or without water-level simulation.
- Each `SensorFrame` carries `water_level_reading`, and each
  `LocalizationStepResult` carries `water_level_reading` and
  `water_level_assessment`; `iter_localization_results()` uses one monitor per
  run so duplicate suppression spans the whole scenario.

Run its focused tests with:

```powershell
python -m unittest tests.test_water_level -v
```

## Streamlit operations dashboard

Install the requirements and launch the current dashboard from the project
root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

The single-page dashboard brings the pipeline view, sensor readings, asset
details, trends, and alerts together. The **Start Simulation** control plays a
deterministic demonstration; pause and reset are available while it runs. The
displayed network geometry and sensor responses are illustrative and do not
represent a surveyed installation or live telemetry.

The dashboard also shows the simulated water level: a headline metric with its
status, a **Water level monitoring** section with the level history, the 200
threshold line, the current status and a table of water-level events, and one
row per continuous water-level event in **Recent alerts** (marked Active or
Cleared). These rows are labelled separately from pipe-segment localization
results.

`streamlit_app.py` also still contains render functions for the earlier
multi-page layout (Overview, Analytics, Alerts, Demo, Requirements, WDSEventDB
Replay); the current entry point renders only the single-page dashboard, and
water level is integrated there.

The existing recorded-data replay and saved inference backend remain separate
from the dashboard simulation. The saved model classifies samples according to
source labels; it has not been validated as a confirmed leak detector, and it
does not provide field-validated localization. SPS does not connect to physical
sensors.

## Before the first commit

1. Resolve the WDSEventDB CC0-versus-CC BY 4.0 scope with the authors; ensure
   dataset attribution is present if required.
2. Confirm whether to publish any dataset files or the `.joblib` model. Current
   ignore rules keep them local; there is no need for Git LFS at their present
   sizes unless repository policy requires it.
3. Review all reports and source references for correctness and decide which
   historical reports should be included. They are distinct, not content
   duplicates.
4. Review the final staged file list for local paths, credentials, private
   data, generated files, or accidental force-added datasets/models.
5. Add a project-level license only after deciding which code and assets it
   covers; no root project license is supplied here.
