"""OpenVINO audio-text late-fusion classification handler."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from src.inference import InferenceHandler, register_handler


@register_handler("openvino_audio_text_classify")
class OpenVINOAudioTextClassifyHandler(InferenceHandler):
    """Classify paired audio and captions using late fusion."""

    def __init__(self):
        self.config = {}
        self.classes = []
        self.audio_compiled = None
        self.text_compiled = None
        self.audio_output = None
        self.text_output = None
        self.audio_scaler = None
        self.text_vectorizer = None

    def can_handle(self, config: dict) -> bool:
        inference_cfg = config.get("inference", {})
        handler = config.get("handler", inference_cfg.get("handler"))
        task = config.get("task", inference_cfg.get("task"))
        return (
            handler == "openvino_audio_text_classify"
            and task == "classify"
        )

    def load(self, config: dict) -> None:
        import openvino as ov
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        self.config = config
        inference_cfg = config.get("inference", {})

        audio_model_path = Path(inference_cfg["audio_model_path"])
        text_model_path = Path(inference_cfg["text_model_path"])
        audio_features_path = Path(inference_cfg["audio_features_path"])
        labels_path = Path(inference_cfg["labels_path"])
        training_manifest_path = Path(
            inference_cfg["training_manifest_path"]
        )
        classes_path = Path(inference_cfg["classes_path"])
        device = config.get(
            "device",
            inference_cfg.get("device", "GPU"),
        )

        required_paths = [
            audio_model_path,
            text_model_path,
            audio_features_path,
            labels_path,
            training_manifest_path,
            classes_path,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"Required Task G file not found: {path}")

        self.classes = json.loads(
            classes_path.read_text(encoding="utf-8")
        )
        if isinstance(self.classes, dict):
            self.classes = [
                self.classes[str(index)]
                for index in range(len(self.classes))
            ]

        audio_features = np.load(audio_features_path).astype(np.float32)
        labels = np.load(labels_path).astype(np.int64)

        import csv

        with training_manifest_path.open(
            encoding="utf-8",
            newline="",
        ) as file:
            training_rows = list(csv.DictReader(file))

        if len(audio_features) != len(labels):
            raise ValueError(
                "audio_features.npy and labels.npy have different lengths"
            )

        if len(training_rows) != len(labels):
            raise ValueError(
                "Training manifest and labels.npy have different lengths"
            )

        indices = np.arange(len(labels))
        train_indices, _ = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )

        self.audio_scaler = StandardScaler().fit(
            audio_features[train_indices]
        )

        # Match train_G.py: TF-IDF was fitted on all retained captions
        # before the train/validation split.
        training_captions = [
            row["caption"]
            for row in training_rows
        ]

        self.text_vectorizer = TfidfVectorizer(
            max_features=128,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self.text_vectorizer.fit(training_captions)

        if len(self.text_vectorizer.vocabulary_) != 128:
            raise ValueError(
                "The fitted TF-IDF vectorizer does not have 128 features"
            )

        core = ov.Core()

        audio_model = core.read_model(str(audio_model_path))
        text_model = core.read_model(str(text_model_path))

        self.audio_compiled = core.compile_model(audio_model, device)
        self.text_compiled = core.compile_model(text_model, device)

        self.audio_output = self.audio_compiled.output(0)
        self.text_output = self.text_compiled.output(0)

        print(
            "✓ Task G models and preprocessing artifacts loaded "
            f"(device: {device})"
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float32)
        shifted = logits - np.max(logits)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum()

    def _extract_audio_features(
        self,
        audio_path: Path,
        sample_rate: int,
        clip_seconds: int,
        n_mfcc: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        import imageio_ffmpeg
        import librosa

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with tempfile.TemporaryDirectory(prefix="task_g_") as temp_dir:
            wav_path = Path(temp_dir) / "audio.wav"
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            subprocess.run(
                [
                    ffmpeg_path,
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(sample_rate),
                    "-t",
                    str(clip_seconds),
                    str(wav_path),
                ],
                check=True,
                capture_output=True,
            )

            waveform, _ = librosa.load(
                wav_path,
                sr=sample_rate,
                mono=True,
                duration=clip_seconds,
            )

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=n_mfcc,
        )

        features = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                [
                    librosa.feature.zero_crossing_rate(
                        y=waveform
                    ).mean(),
                    librosa.feature.spectral_rolloff(
                        y=waveform,
                        sr=sample_rate,
                    ).mean(),
                    librosa.feature.spectral_centroid(
                        y=waveform,
                        sr=sample_rate,
                    ).mean(),
                    librosa.feature.rms(y=waveform).mean(),
                ],
            ]
        ).astype(np.float32)

        if features.shape != (30,):
            raise ValueError(
                f"Expected 30 audio features, received {features.shape}"
            )

        return waveform.astype(np.float32), features

    @staticmethod
    def _compact_waveform(
        waveform: np.ndarray,
        max_points: int,
    ) -> list[float]:
        if len(waveform) <= max_points:
            compact = waveform
        else:
            indices = np.linspace(
                0,
                len(waveform) - 1,
                max_points,
                dtype=np.int64,
            )
            compact = waveform[indices]

        return compact.astype(float).tolist()

    def infer(self, inputs: list, config: dict) -> list[dict]:
        import csv

        merged_config = {**self.config, **config}
        inference_cfg = merged_config.get("inference", {})

        manifest_path = Path(inference_cfg["manifest_path"])
        audio_dir = Path(inference_cfg["audio_dir"])
        source_column = inference_cfg.get("source_column", "file_name")
        caption_column = inference_cfg.get("caption_column", "caption")

        sample_rate = int(inference_cfg.get("sample_rate", 16000))
        clip_seconds = int(inference_cfg.get("clip_seconds", 10))
        n_mfcc = int(inference_cfg.get("n_mfcc", 13))
        max_points = int(
            inference_cfg.get("raw_waveform_max_points", 1000)
        )

        num_samples = merged_config.get("num_images")
        fusion_weights = inference_cfg.get(
            "fusion_weights",
            {"audio": 0.5, "text": 0.5},
        )
        audio_weight = float(fusion_weights.get("audio", 0.5))
        text_weight = float(fusion_weights.get("text", 0.5))

        with manifest_path.open(
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        if num_samples:
            rows = rows[: int(num_samples)]

        results = []

        print(f"Running Task G inference on {len(rows)} samples...")

        for index, row in enumerate(rows):
            source = row[source_column]
            caption = row[caption_column]
            audio_path = audio_dir / source

            waveform, raw_audio_features = self._extract_audio_features(
                audio_path,
                sample_rate,
                clip_seconds,
                n_mfcc,
            )

            normalized_audio = self.audio_scaler.transform(
                raw_audio_features.reshape(1, -1)
            ).astype(np.float32)

            text_features = self.text_vectorizer.transform(
                [caption]
            ).toarray().astype(np.float32)

            audio_logits = np.asarray(
                self.audio_compiled([normalized_audio])[self.audio_output]
            )[0]

            text_logits = np.asarray(
                self.text_compiled([text_features])[self.text_output]
            )[0]

            fused_logits = (
                audio_weight * audio_logits
                + text_weight * text_logits
            )

            audio_probabilities = self._softmax(audio_logits)
            text_probabilities = self._softmax(text_logits)
            fused_probabilities = self._softmax(fused_logits)

            label_id = int(np.argmax(fused_probabilities))
            label = self.classes[label_id]
            confidence = float(fused_probabilities[label_id])

            raw_signal = {
                "sample_rate": sample_rate,
                "num_samples": int(len(waveform)),
                "waveform": self._compact_waveform(
                    waveform,
                    max_points,
                ),
            }
            feature_evidence = {
                "raw": raw_audio_features.astype(float).tolist(),
                "normalized": normalized_audio[0].astype(float).tolist(),
            }

            sample_id = row.get("encord_audio_id", index)

            results.append(
                {
                    "sample_id": int(sample_id),
                    "source": source,
                    "label": label,
                    "label_id": label_id,
                    "confidence": confidence,
                    "probabilities": fused_probabilities.astype(
                        float
                    ).tolist(),
                    "audio_confidence": float(
                        np.max(audio_probabilities)
                    ),
                    "text_confidence": float(
                        np.max(text_probabilities)
                    ),
                    "sensor_type": "audio",
                    "sensor_raw_json": json.dumps(raw_signal),
                    "sensor_features_json": json.dumps(
                        feature_evidence
                    ),
                    "text_caption": caption,
                }
            )

            print(
                f"[{index + 1}/{len(rows)}] "
                f"{source} -> {label} ({confidence:.4f})"
            )

        print(f"✓ Task G inference completed ({len(results)} samples)")
        return results

    def get_output_schema(self) -> dict:
        return self.config.get("schema", {})