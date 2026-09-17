# Water Treatment Use Case

## Use Case Overview

The Water Treatment use case classifies paired satellite raster patches and
environmental text prompts by irrigation type. It runs separate OpenVINO image
and text branches and combines their outputs through late fusion.

This is a **multimodal classification** use case. It does not perform object
detection and does not produce bounding boxes.

The four output classes are:

| Class ID | Class name |
|---:|---|
| 0 | `non_irrigated` |
| 1 | `flood` |
| 2 | `sprinkler` |
| 3 | `drip` |

These labels describe predicted irrigation categories for individual raster
patches. A classification should not be treated as confirmation of field
conditions without additional geographic or operational evidence.

## Dataset

The source data comes from the IRRISIGHT dataset. For this integration, 500
Arizona patches were streamed and cached locally. Each cached sample consists
of:

- a multichannel raster patch stored as a NumPy `.npy` file;
- a JSON metadata record containing the state, label, label type, label
  confidence, and environmental text prompt.

The training script applied the configured confidence filter and per-class cap
to build a curated 121-sample manifest:

| Class | Curated samples |
|---|---:|
| `flood` | 50 |
| `sprinkler` | 50 |
| `drip` | 14 |
| `non_irrigated` | 7 |
| **Total** | **121** |

The curated manifest is generated locally by the training script. It pairs each raster patch with its state, class label, and text prompt.

The final end-to-end pipeline check processed the first 100 manifest records using `--num-images 100`.

The curated dataset remains imbalanced because the downloaded Arizona subset contains substantially fewer `drip` and `non_irrigated` samples than `flood` and `sprinkler` samples. The per-class cap limits the majority classes but cannot create additional minority-class observations.

## Model and Preprocessing

### Image preprocessing

Each multichannel raster patch is converted into 60 handcrafted features. For each of the 15 channels, the handler calculates:

- mean;
- standard deviation;
- 10th percentile;
- 90th percentile.

The resulting vector is normalized using the scaler stored in `preprocessing.joblib`.

### Image branch

- **Model type:** multi-layer perceptron (MLP)
- **Input:** image feature vector `[?, 60]`
- **Output:** irrigation class logits `[?, 4]`
- **Runtime:** OpenVINO

### Text preprocessing

+ The environmental text prompt associated with each patch is transformed into a 128-dimensional TF-IDF vector using the fitted vectorizer stored in `preprocessing.joblib`.

### Text branch

- **Model type:** multi-layer perceptron (MLP)
- **Input:** TF-IDF text feature vector `[?, 128]`
- **Output:** irrigation class logits `[?, 4]`
- **Runtime:** OpenVINO

### Shared output classes

- `non_irrigated`
- `flood`
- `sprinkler`
- `drip`

### Late fusion

The fusion weights stored in `preprocessing.joblib` are:

```yaml
fusion_weights:
  image: 0.5
  text: 0.5
```

The image and text branches process their paired inputs independently. Their weighted logits are combined, and softmax is applied to produce the fused class probabilities. The pipeline selects the class with the highest fused probability.


## OpenVINO Image/Text Classification Handler

The use case is handled by:

```text
src/inference/handlers/openvino_image_text_classify.py
```

The handler:

1. validates the OpenVINO and preprocessing artifacts;
2. loads the scaler, vectorizer, class order, and fusion weights;
3. compiles both OpenVINO models on the requested device;
4. reads the raster patch and paired text prompt from the manifest;
5. prepares the 60-dimensional image input and 128-dimensional text input;
6. runs both model branches;
7. performs late fusion and returns fused and branch confidence values; and
8. optionally generates a displayable JPEG preview for ticket creation.

## Pipeline Flow

```text
Raster patch ─> 60 statistical features ─> StandardScaler ─> image.xml ─┐
                                                                          ├─> late fusion ─> class + confidence
Text prompt ──> 128 TF-IDF features ───────────────────────> text.xml ──┘
```

The fused classifications are written to JSONL and SQLite. The policy,
analysis, and evidence agents then process the stored results.

## Configuration

The use-case configuration is located at:

```text
config/water_treatment/config.yaml
```

The essential inference settings are:

```yaml
modality: multi

inference:
  handler: openvino_image_text_classify
  task: classify
  device: GPU
  input_mode: multi
  ticket_images: true
  preview_channels: [0, 1, 2]
  preview_percentiles: [2, 98]
  image_model_path: models/ov_models/water_treatment/image.xml
  text_model_path: models/ov_models/water_treatment/text.xml
  preprocessing_path: models/ov_models/water_treatment/preprocessing.joblib
  manifest_path: datasets/water_treatment/manifest.csv
```

The preview channel selection and contrast percentiles are configuration
values rather than Task H constants in the shared ticket code. The generated
preview is a display-oriented three-channel rendering. Its colors should not
be interpreted as calibrated natural color unless the configured channels are
known to correspond to the appropriate visible bands.

The fallback policy uses a global threshold of 0.5 and a threshold of 0.5 for
all four classes. When model-based policy generation is enabled, the policy
agent may produce different per-class thresholds. The values actually applied
to a run are stored in `out/water_treatment/agent/policy.json`.

## SQLite Schema

Each row represents one fused image/text classification.

| Column | Type | Description |
|---|---|---|
| `image_id` | INTEGER | Manifest-order sample identifier |
| `source` | TEXT | Raster patch key |
| `state` | TEXT | Source state, such as Arizona |
| `label` | TEXT | Fused predicted class |
| `confidence` | REAL | Fused prediction confidence |
| `image_confidence` | REAL | Maximum image-branch probability |
| `text_confidence` | REAL | Maximum text-branch probability |
| `image_features_json` | TEXT | Raw and normalized image features |
| `text_prompt` | TEXT | Environmental prompt used by the text branch |
| `patch_path` | TEXT | Local path to the source raster patch |
| `created_at` | TIMESTAMP | Record insertion time |

## Agent Outputs

The workflow uses the policy, analysis, and evidence agents.

- The policy agent defines global and per-class confidence thresholds.
- The analysis agent summarizes accepted classes and fused, image, and text
  confidence statistics.
- The evidence agent records filtering totals and representative records.

Generated files include:

```text
out/water_treatment/agent/policy.json
out/water_treatment/agent/analysis_report.json
out/water_treatment/agent/analysis_summary.txt
out/water_treatment/agent/evidence.json
out/water_treatment/agent/evidence_trail.txt
```

In one verified 100-record run, the generated policy accepted 14 records and
filtered 86. These values are run-specific because model-based policy
generation may change the per-class thresholds.

## Model Evaluation and Limitations

The selected training run reported an overall validation accuracy of 0.72:

| Class | Validation accuracy |
|---|---:|
| `non_irrigated` | 0.50 |
| `flood` | 0.80 |
| `sprinkler` | 0.60 |
| `drip` | 1.00 |

These per-class values are based on a small stratified validation split. In
particular, the minority-class measurements contain few validation examples
and should not be treated as stable production estimates.

The model also agreed with 98 of the 121 curated manifest labels in a full
manifest inference check. That 80.99% agreement is **not independent test
accuracy**, because the same curated dataset contributed to training. It is an
integration and consistency check only.

Further evaluation should use a separate held-out state or dataset with enough
examples of all four classes.



## Running the Use Case

### Run inference

```bash
python run_inference_oep.py \
  --config config/water_treatment/config.yaml \
  --device GPU \
  --num-images 100
```

Although the shared option is named `--num-images`, Task H interprets it as the
maximum number of paired manifest samples.

### Run agent orchestration

```bash
python -m scripts.run_agent_orchestration \
  --use-case water_treatment
```

### Run interactive CLI chat

```bash
python interactive_chat.py \
  --config config/water_treatment/config.yaml
```

Example database questions include:

```text
How many records in Arizona have the label flood?
Show database records where the label is drip and confidence is above 0.8.
Using WHERE label IN ('sprinkler', 'flood', 'non_irrigated'), count database records grouped by label.
Show me the text prompt for patch #42.
```

Using exact stored class labels in database questions helps the SQL model
generate reliable filters.

### Run the Web UI

```bash
python scripts/launch_web_app.py start
```

Select **Water Treatment**, choose **Multi-modal**, select the device, and run
the pipeline.

## Web UI and Ticket Verification

The following flow was verified:

1. Run the pipeline for 100 paired samples.
2. Run the policy, analysis, and evidence workflow.
3. Query the stored classifications in Web UI chat.
4. Request the patch with the highest fused confidence.
5. Create a ticket from the returned row.
6. Open the ticket and verify its classification, evidence trail, and image
   preview.

The verified ticket for image ID `64` displayed:

- predicted label: `non_irrigated`;
- fused confidence: approximately `0.858`;
- the configured raster preview;
- the corresponding agent evidence trail.

The handler writes each preview to the runtime output directory using the
source key as the filename stem. The shared ticket agent then locates and
embeds that JPEG without any Task H-specific path or raster logic.

## Device Status

Task H inference was verified with both OpenVINO branches compiled on the Intel
integrated GPU. CPU is also available as an OpenVINO execution device, but the
reported Task H validation was performed on the Intel integrated GPU.

## Summary

The Water Treatment use case provides end-to-end image/text irrigation
classification with OpenVINO. It pairs multichannel raster statistics with
TF-IDF environmental text features, performs two-branch late fusion, stores
modality-specific evidence in JSONL and SQLite, supports agent reporting and
natural-language database queries, and creates tickets containing configured
raster previews without adding Task H-specific behavior to the shared ticket
system.
