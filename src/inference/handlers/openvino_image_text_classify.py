"""OpenVINO image-text late-fusion classification handler."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.inference import InferenceHandler, register_handler


@register_handler("openvino_image_text_classify")
class OpenVINOImageTextClassifyHandler(InferenceHandler):
    """Classify paired raster patches and text prompts using late fusion."""

    def __init__(self):
        self.config = {}
        self.classes = []
        self.image_compiled = None
        self.text_compiled = None
        self.image_output = None
        self.text_output = None
        self.image_scaler_mean = None
        self.image_scaler_scale = None
        self.text_vectorizer = None
        self.fusion_weights = {}

    def can_handle(self, config: dict) -> bool:
        inference_cfg = config.get("inference", {})
        handler = config.get("handler", inference_cfg.get("handler"))
        task = config.get("task", inference_cfg.get("task"))
        return handler == "openvino_image_text_classify" and task == "classify"

    def load(self, config: dict) -> None:
        import openvino as ov
        import joblib

        self.config = config
        inference_cfg = config.get("inference", {})

        image_model_path = Path(inference_cfg["image_model_path"])
        text_model_path = Path(inference_cfg["text_model_path"])
        preprocessing_path = Path(inference_cfg["preprocessing_path"])
        device = config.get("device", inference_cfg.get("device", "GPU"))

        for path in (
            image_model_path,
            text_model_path,
            preprocessing_path,
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required Task H file not found: {path}"
                )

        checkpoint = joblib.load(preprocessing_path)

        self.classes = list(checkpoint["classes"])
        self.image_scaler_mean = np.asarray(
            checkpoint["image_scaler_mean"],
            dtype=np.float32,
        )
        self.image_scaler_scale = np.asarray(
            checkpoint["image_scaler_scale"],
            dtype=np.float32,
        )
        self.text_vectorizer = checkpoint["text_vectorizer"]
        self.fusion_weights = checkpoint.get(
            "fusion_weights",
            {"image": 0.5, "text": 0.5},
        )

        if self.image_scaler_mean.shape != (60,):
            raise ValueError("Expected a 60-dimensional image scaler")

        if len(self.text_vectorizer.vocabulary_) != 128:
            raise ValueError("Expected a 128-dimensional TF-IDF vectorizer")

        core = ov.Core()
        image_model = core.read_model(str(image_model_path))
        text_model = core.read_model(str(text_model_path))

        self.image_compiled = core.compile_model(image_model, device)
        self.text_compiled = core.compile_model(text_model, device)
        self.image_output = self.image_compiled.output(0)
        self.text_output = self.text_compiled.output(0)

        print(
            "✓ Task H models and preprocessing artifacts loaded "
            f"(device: {device})"
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    @staticmethod
    def _extract_image_features(patch: np.ndarray) -> np.ndarray:
        if patch.ndim < 2:
            raise ValueError(
                f"Expected a multi-channel raster patch, received {patch.shape}"
            )

        flat = patch.reshape(patch.shape[0], -1).astype(np.float32)
        features = np.concatenate(
            [
                flat.mean(axis=1),
                flat.std(axis=1),
                np.percentile(flat, 10, axis=1),
                np.percentile(flat, 90, axis=1),
            ]
        ).astype(np.float32)

        if features.shape != (60,):
            raise ValueError(
                f"Expected 60 image features, received {features.shape}"
            )

        return features

    @staticmethod
    def _save_preview(
        patch: np.ndarray,
        output_path: Path,
        channels: list[int],
        percentiles: list[float],
    ) -> None:
        """Save a configured three-channel raster preview as JPEG."""
        if patch.ndim != 3:
            raise ValueError(
                f"Preview requires a 3D raster patch, received {patch.shape}"
            )

        if len(channels) != 3:
            raise ValueError(
                "preview_channels must contain exactly three channel indexes"
            )

        if len(percentiles) != 2:
            raise ValueError(
                "preview_percentiles must contain low and high values"
            )

        if any(channel < 0 or channel >= patch.shape[0] for channel in channels):
            raise ValueError(
                f"Preview channels {channels} are invalid for {patch.shape}"
            )

        low_percentile, high_percentile = map(float, percentiles)
        if not 0 <= low_percentile < high_percentile <= 100:
            raise ValueError(
                "preview_percentiles must satisfy 0 <= low < high <= 100"
            )

        selected = patch[np.asarray(channels)].astype(np.float32)
        preview = np.zeros(selected.shape, dtype=np.uint8)

        for index, channel in enumerate(selected):
            finite = channel[np.isfinite(channel)]
            if finite.size == 0:
                continue

            low, high = np.percentile(
                finite,
                [low_percentile, high_percentile],
            )

            if high <= low:
                high = low + 1.0

            scaled = np.clip(
                (channel - low) / (high - low),
                0.0,
                1.0,
            )
            scaled = np.nan_to_num(scaled)
            preview[index] = (scaled * 255).astype(np.uint8)

        rgb = np.moveaxis(preview, 0, -1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(
            output_path,
            format="JPEG",
            quality=90,
        )

    def infer(self, inputs: list, config: dict) -> list[dict]:
        merged_config = {**self.config, **config}
        inference_cfg = merged_config.get("inference", {})

        manifest_path = Path(inference_cfg["manifest_path"])
        num_samples = merged_config.get("num_images")

        with manifest_path.open(
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        if num_samples:
            rows = rows[: int(num_samples)]

        image_weight = float(self.fusion_weights.get("image", 0.5))
        text_weight = float(self.fusion_weights.get("text", 0.5))

        ticket_images = bool(
            inference_cfg.get("ticket_images", False)
        )
        preview_channels = list(
            inference_cfg.get("preview_channels", [])
        )
        preview_percentiles = list(
            inference_cfg.get("preview_percentiles", [])
        )

        viz_dir = None
        if ticket_images:
            output_file = merged_config.get("output_file")
            if not output_file:
                raise ValueError(
                    "ticket_images requires an output_file"
                )

            viz_dir = Path(output_file).parent / "viz"
            viz_dir.mkdir(parents=True, exist_ok=True)

        results = []
        print(f"Running Task H inference on {len(rows)} samples...")

        for index, row in enumerate(rows):
            patch_path = Path(row["patch_path"])
            if not patch_path.exists():
                raise FileNotFoundError(
                    f"Raster patch not found: {patch_path}"
                )

            patch = np.load(patch_path)

            if ticket_images and viz_dir is not None:
                source_stem = Path(str(row["key"])).stem
                if not source_stem:
                    raise ValueError("Manifest key cannot be empty")

                self._save_preview(
                    patch=patch,
                    output_path=viz_dir / f"{source_stem}.jpg",
                    channels=preview_channels,
                    percentiles=preview_percentiles,
                )

            raw_image_features = self._extract_image_features(patch)

            normalized_image = (
                (
                    raw_image_features - self.image_scaler_mean
                )
                / self.image_scaler_scale
            ).reshape(1, -1).astype(np.float32)

            text_prompt = row["text_prompt"]
            text_features = self.text_vectorizer.transform(
                [text_prompt]
            ).toarray().astype(np.float32)

            image_logits = np.asarray(
                self.image_compiled([normalized_image])[self.image_output]
            )[0]
            text_logits = np.asarray(
                self.text_compiled([text_features])[self.text_output]
            )[0]

            fused_logits = (
                image_weight * image_logits
                + text_weight * text_logits
            )

            image_probabilities = self._softmax(image_logits)
            text_probabilities = self._softmax(text_logits)
            fused_probabilities = self._softmax(fused_logits)

            label_id = int(np.argmax(fused_probabilities))
            label = self.classes[label_id]
            confidence = float(fused_probabilities[label_id])

            results.append(
                {
                    "image_id": index,
                    "source": row["key"],
                    "state": row["state"],
                    "label": label,
                    "label_id": label_id,
                    "confidence": confidence,
                    "probabilities": fused_probabilities.astype(
                        float
                    ).tolist(),
                    "image_confidence": float(
                        np.max(image_probabilities)
                    ),
                    "text_confidence": float(
                        np.max(text_probabilities)
                    ),
                    "image_features_json": json.dumps(
                        {
                            "raw": raw_image_features.astype(float).tolist(),
                            "normalized": normalized_image[0]
                            .astype(float)
                            .tolist(),
                        }
                    ),
                    "text_prompt": text_prompt,
                    "patch_path": str(patch_path),
                }
            )

            print(
                f"[{index + 1}/{len(rows)}] "
                f"{row['key']} -> {label} ({confidence:.4f})"
            )

        print(f"✓ Task H inference completed ({len(results)} samples)")
        return results

    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})