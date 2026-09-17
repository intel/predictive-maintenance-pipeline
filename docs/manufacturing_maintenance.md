# Manufacturing Maintenance Use Case

## Use Case Overview

The Manufacturing Maintenance use case classifies paired audio clips and text captions into acoustic event categories. It uses two OpenVINO models—one for audio features and one for TF-IDF text features—and combines their outputs through late fusion.

This is a **multimodal classification** use case. It does not perform object detection and does not produce bounding boxes.

The five output classes are:

| Class ID | Class name |
|---:|---|
| 0 | `alert_signal` |
| 1 | `ambient_normal` |
| 2 | `impact_event` |
| 3 | `rotating_machinery` |
| 4 | `vehicle_engine` |

`ambient_normal` represents normal acoustic conditions. The other labels describe acoustic event categories; they should not be interpreted as confirmed equipment failures without additional evidence.

## Dataset

The local test subset contains 100 paired audio-caption samples:

- Manifest: `datasets/manufacturing_maintenance/manifest_balanced.csv`
- Audio clips: `datasets/manufacturing_maintenance/audio/`
- Audio format: MP4
- Clip duration: up to 10 seconds
- Text input: the caption associated with each audio clip

The manifest provides the source filename, caption, and sample metadata used to pair the two modalities.

## Models and Preprocessing Artifacts

Two supplied OpenVINO IR models are used:

| Branch | Model | Input | Output |
|---|---|---:|---:|
| Audio | `models/ov_models/manufacturing_maintenance/audio.xml` | 30 features | 5 class scores |
| Text | `models/ov_models/manufacturing_maintenance/text.xml` | 128 TF-IDF features | 5 class scores |

Each XML file requires its corresponding BIN file.

The supplied preprocessing artifacts are stored in the same model directory:

- `audio_features.npy`: 263 training audio-feature vectors with shape `(263, 30)`
- `labels.npy`: 263 training labels with shape `(263,)`
- `manifest.csv`: 263 training filenames, captions, and labels
- `classes.json`: the ordered list of five class names

These artifacts are required because the OpenVINO models accept prepared feature vectors rather than raw MP4 files or raw text.

### Audio preprocessing

Each MP4 file is decoded with the FFmpeg binary supplied by `imageio-ffmpeg`, converted to mono audio at 16 kHz, and limited to 10 seconds. The handler then extracts 30 features:

- mean and standard deviation of 13 MFCC coefficients: 26 values
- mean zero-crossing rate: 1 value
- mean spectral rolloff: 1 value
- mean spectral centroid: 1 value
- mean RMS energy: 1 value

The resulting vector is normalized with a `StandardScaler`. The scaler is reconstructed from the supplied training features using the same stratified 80/20 split and `random_state=42` used by the training script.

### Text preprocessing

The caption is transformed into a 128-dimensional TF-IDF vector. The fitted vocabulary is reconstructed from the supplied training manifest with:

- `max_features=128`
- English stop words
- unigram and bigram features
- sublinear term-frequency scaling
- `min_df=1`

Reconstructing both preprocessors ensures that inference inputs follow the training-time feature preparation.

## OpenVINO Audio/Text Classification Handler

The use case is handled by:

```text
src/inference/handlers/openvino_audio_text_classify.py
```

The handler:

1. validates the model, class, and preprocessing artifacts;
2. reconstructs the audio scaler and text vectorizer;
3. compiles both OpenVINO models on the requested device;
4. preprocesses each audio-caption pair;
5. runs the audio and text branches;
6. combines the branch outputs using configured fusion weights; and
7. returns the fused label, fused confidence, branch confidences, caption, and audio evidence.

The default fusion weights are:

```yaml
fusion_weights:
  audio: 0.5
  text: 0.5
```

## Pipeline Flow

```text
MP4 audio ──> 30 acoustic features ──> StandardScaler ──> audio.xml ──┐
                                                                       ├─> late fusion ─> class + confidence
Caption ───> 128 TF-IDF features ───────────────────────> text.xml ───┘
```

The fused classification is written to JSONL and SQLite. The policy, analysis, and evidence agents then process the stored results.

## Configuration

The use-case configuration is located at:

```text
config/manufacturing_maintenance/config.yaml
```

The essential inference settings are:

```yaml
modality: multi

inference:
  handler: openvino_audio_text_classify
  task: classify
  device: GPU
  input_mode: multi
  audio_model_path: models/ov_models/manufacturing_maintenance/audio.xml
  text_model_path: models/ov_models/manufacturing_maintenance/text.xml
  manifest_path: datasets/manufacturing_maintenance/manifest_balanced.csv
  audio_dir: datasets/manufacturing_maintenance/audio
  training_manifest_path: models/ov_models/manufacturing_maintenance/manifest.csv
  audio_features_path: models/ov_models/manufacturing_maintenance/audio_features.npy
  labels_path: models/ov_models/manufacturing_maintenance/labels.npy
  classes_path: models/ov_models/manufacturing_maintenance/classes.json
```

The Web UI presents this mode as **Audio + Text**.

The fallback policy uses a global threshold of 0.5 and a threshold of 0.5 for every class. When model-based policy generation is enabled, the generated policy may select different per-class thresholds; the applied values are saved to `out/manufacturing_maintenance/agent/policy.json`.

## SQLite Schema

Inference results are stored in the `detections` table. Important columns are:

| Column | Type | Description |
|---|---|---|
| `sample_id` | INTEGER | Source sample identifier |
| `source` | TEXT | Source MP4 filename |
| `label` | TEXT | Fused predicted class |
| `confidence` | REAL | Fused prediction confidence |
| `audio_confidence` | REAL | Maximum audio-branch probability |
| `text_confidence` | REAL | Maximum text-branch probability |
| `sensor_type` | TEXT | Audio sensor modality |
| `sensor_raw_json` | TEXT | Raw audio metadata or sampled waveform evidence |
| `sensor_features_json` | TEXT | Extracted and normalized audio features |
| `text_caption` | TEXT | Caption used by the text branch |
| `created_at` | TIMESTAMP | Record insertion time |

## Agent Outputs

The configured workflow uses the policy, analysis, and evidence agents.

- The policy agent defines the global and per-class confidence thresholds.
- The analysis agent summarizes accepted classifications and audio/text confidence statistics.
- The evidence agent records filtering totals and representative samples with their source, caption, fused confidence, and branch confidences.

Generated files include:

```text
out/manufacturing_maintenance/agent/policy.json
out/manufacturing_maintenance/agent/analysis_report.json
out/manufacturing_maintenance/agent/analysis_summary.txt
out/manufacturing_maintenance/agent/evidence.json
out/manufacturing_maintenance/agent/evidence_trail.txt
```

## Test-Set Evaluation

The complete 100-sample local subset was successfully processed on an Intel integrated GPU:

- JSONL records: 100
- SQLite rows: 100
- Audio and text inputs were both evaluated for every sample
- Fused predictions and branch confidence values were stored successfully

In the reproduced 53-sample validation split, the supplied fused model achieved 22 correct predictions (41.51%). This figure describes the supplied model and reconstructed training-time preprocessing; it is not a benchmark of the orchestration or Web UI components.

The tested orchestration run accepted 24 of 100 classifications under its generated policy and filtered 76. Because policy generation can produce different thresholds, these counts are run-specific.

## Output Artifacts

The main inference outputs are:

```text
out/manufacturing_maintenance/detections.jsonl
out/manufacturing_maintenance/sql_data/detections.db
```

Task G has no image modality. Therefore, ticket generation includes the classification and evidence trail but intentionally omits the image section.

## Running the Use Case

### Run inference

```bash
python run_inference_oep.py \
  --config config/manufacturing_maintenance/config.yaml \
  --device GPU \
  --num-images 100
```

Although the common CLI option is named `--num-images`, Task G interprets it as the maximum number of manifest samples.

### Run agent orchestration

```bash
python -m scripts.run_agent_orchestration \
  --use-case manufacturing_maintenance
```

### Run interactive CLI chat

```bash
python interactive_chat.py \
  --config config/manufacturing_maintenance/config.yaml
```

Example database question:

```text
How many alert_signal events were classified?
```

The verified run generated:

```sql
SELECT COUNT(*) FROM detections WHERE label = 'alert_signal';
```

and returned 7 records.

### Run the Web UI

```bash
python scripts/launch_web_app.py start
```

Select **Manufacturing Maintenance**, choose **Audio + Text**, select the device, and run the pipeline.

## Web UI and Ticket Verification

The following flow was verified in the Web UI:

1. Run inference for the 100 paired samples.
2. Run the agent workflow.
3. Query the SQLite results through chat.
4. Select a returned row that contains `sample_id`.
5. Create and open its ticket.

The verified ticket for sample `142975` displayed:

- source: `rZkeCI687sI_30.mp4`
- label: `rotating_machinery`
- fused confidence: approximately `0.969`
- the evidence trail
- no image section, as expected for an audio/text use case

## Device Status

Task G inference was verified on the Intel integrated GPU. CPU is also available through OpenVINO. NPU execution was not validated as part of this test and should not be considered confirmed.

## Summary

The Manufacturing Maintenance use case provides end-to-end audio/text classification with OpenVINO. It reconstructs the required training-time preprocessors, performs two-branch late fusion, stores modality-specific evidence in JSONL and SQLite, supports agent analysis and natural-language database queries, and creates sensor-only result tickets without an image section.
