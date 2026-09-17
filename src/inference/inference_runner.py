"""Top-level inference runner used by the CLI entry point."""

from pathlib import Path

from scripts.clean_slate import clean_database
from src.inference import dispatch
from src.inference.config_builder import build_handler_config
from src.inference.output_writer import write_inference_outputs
from src.inference.visualization import (
    generate_classification_viz,
    generate_detection_viz,
)


# Import SQLite client
try:
    from src.utility.sqlite_client import SQLiteClient
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False
    print("SQLite client not available.")


def run_inference(model_path, model_proc_path, images_dir, output_file,
                  device='GPU', conf_threshold=0.25, num_images=None,
                  nireq=4, sqlite_db_path=None, clear_on_run=True,
                  video_path=None, inference_interval=1,
                  schema=None, detection_mapping=None, out_subdir='',
                  task='detect', modality='image', class_names=None,
                  sensor_model_path=None, sensor_data_path=None,
                  fusion_weights=None, handler_name=None, config=None):
    """Run full inference pipeline: DLStreamer → parse → JSONL + SQLite.

    Supports modality-aware inference:
    - 'image': DLStreamer only (detect or classify)
    - 'sensor': Sensor MLP only
    - 'multi': DLStreamer + Sensor MLP + Late Fusion (weighted avg)

    Args:
        schema: Schema dict from config (passed to SQLiteClient).
        detection_mapping: Dict mapping column names to dot-paths in detection objects.
        out_subdir: Subdirectory under out/ for this use case.
        task: 'detect' or 'classify'
        modality: 'image', 'sensor', or 'multi'
        class_names: Dict mapping class_id -> class_name (for classification)
        sensor_model_path: Path to sensor MLP model XML (for sensor/multi modality)
        sensor_data_path: Path to sensor CSV data file
        fusion_weights: Dict with 'image' and 'sensor' weights for late fusion
    """

    # Initialize SQLite client
    sqlite_client = None
    if SQLITE_AVAILABLE and sqlite_db_path:
        try:
            sqlite_client = SQLiteClient(db_path=sqlite_db_path, schema=schema)
            print(f"SQLite client connected to {sqlite_db_path}")
        except Exception as e:
            print(f"⚠️  Failed to connect to SQLite: {e}")
            print("   Continuing with JSONL output only...")

    use_video = video_path and Path(video_path).exists()
    images_path = Path(images_dir) if images_dir else None
    image_files = []
    image_names = []
    if images_path and not use_video:
        all_images = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        if num_images:
            all_images = all_images[:num_images]
        image_files = all_images
        image_names = [f.name for f in image_files]

    # Default fusion weights
    if not fusion_weights:
        fusion_weights = {'image': 0.5, 'sensor': 0.5}

        if config is None:
            config = {
                "task": task,
                "modality": modality,
                "names": class_names or {},
                "fusion_weights": fusion_weights,
                "inference": {
                    "handler": handler_name,
                    "task": task,
                    "model_path": model_path,
                    "model_proc_path": model_proc_path,
                    "images_path": images_dir,
                    "output_file": output_file,
                    "device": device,
                    "confidence_threshold": conf_threshold,
                    "num_images": num_images,
                    "nireq": nireq,
                    "video_path": video_path,
                    "inference_interval": inference_interval,
                    "out_subdir": out_subdir,
                },
                "sensor": {
                    "model_path": sensor_model_path,
                    "data_path": sensor_data_path,
                },
            }

    handler_config = build_handler_config(
        config=config,
        schema=schema,
        image_names=image_names,
        output_file=output_file,
        out_subdir=out_subdir,
    )
    # Command-line/runtime device overrides the configured default.
    handler_config["device"] = device
    handler_config["inference"]["device"] = device
    handler_config["num_images"] = num_images

    handler = dispatch(handler_config)
    handler.load(handler_config)
    results = handler.infer(image_names, handler_config)
    output_result = write_inference_outputs(
        results=results,
        handler=handler,
        handler_config=handler_config,
        sqlite_client=sqlite_client,
        sqlite_db_path=sqlite_db_path,
        clear_on_run=clear_on_run,
        clean_database_fn=clean_database,
        detection_mapping=detection_mapping,
        image_files=image_files,
        visualization_fn=generate_classification_viz,
        detection_visualization_fn=generate_detection_viz,
    )

    output_path = output_result["output_path"]
    total_detections = output_result["total_detections"]
    n_frames = output_result["n_frames"]
    print(f"\n{'='*60}")
    print(f"Inference Complete ({modality} / {task})")
    print(f"{'='*60}")
    if task == 'detect':
        processed_label = 'Frames'
    elif modality == 'sensor':
        processed_label = 'Sensor samples'
    elif modality == 'multi':
        processed_label = 'Multi-modal samples'
    elif modality == 'image':
        processed_label = 'Images'
    else:
        processed_label = 'Samples'

    print(f"{processed_label} processed: {n_frames}")
    print(f"Total {'detections' if task == 'detect' else 'classifications'}: {total_detections}")
    if task == 'detect':
        print(f"Average per frame: {total_detections/max(n_frames,1):.1f}")
    print(f"JSONL output: {output_path}")
    if sqlite_client:
        print(f"SQLite output: {sqlite_db_path}")
    if use_video:
        out_base = Path('out') / out_subdir if out_subdir else Path('out')
        annotated = out_base / 'annotated_output.mp4'
        if annotated.exists():
            print(f"Annotated video: {annotated}")
        viz_dir = out_base / 'viz'
        if viz_dir.exists():
            viz_count = len(list(viz_dir.glob('frame_*.jpg')))
            if viz_count:
                print(
                    f"Annotated frames: {viz_dir}/ "
                    f"({viz_count} images, frame_NNNNNN.jpg matches SQL frame_id)"
                )
    else:
        out_base = Path('out') / out_subdir if out_subdir else Path('out')
        viz_dir = out_base / 'viz'
        if viz_dir.exists():
            viz_count = len(list(viz_dir.glob('*.jpg')))
            if viz_count:
                print(f"Annotated images: {viz_dir}/ ({viz_count} images)")
