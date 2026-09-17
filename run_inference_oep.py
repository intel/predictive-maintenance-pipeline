#!/usr/bin/env python3
"""
DLStreamer (OpenVINO Execution Provider) inference for YOLO.
Uses Intel DLStreamer via Docker for GPU-accelerated inference pipeline.
Writes results to JSONL file and SQLite database.
Uses gvawatermark for video visualization output.

Use-case-specific configuration (class names, SQL schema, detection mapping)
is loaded from config/<use_case_id>.yaml.
"""

import json
import yaml
from pathlib import Path
import argparse
from scripts.clean_slate import clean_outputs
from src.inference.inference_runner import run_inference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DLStreamer (OEP) inference for YOLO with SQLite output')
    parser.add_argument('--model', default=None,
                        help='Path to OpenVINO model XML (default: from config)')
    parser.add_argument('--model-proc', default=None,
                        help='Path to model_proc.json (default: alongside model)')
    parser.add_argument('--images', default=None,
                        help='Directory containing images (default: from config)')
    parser.add_argument('--video', default=None,
                        help='Path to input video file (overrides --images)')
    parser.add_argument('--output', default=None,
                        help='Output JSONL file (default: out/<use-case-id>/detections.jsonl)')
    parser.add_argument('--device', default=None,
                        help='OpenVINO inference device, e.g. CPU, GPU, or NPU')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Limit number of images to process (image mode only)')
    parser.add_argument('--inference-interval', type=int, default=1,
                        help='Run inference every Nth frame (video mode, DLStreamer native, default: 1)')
    parser.add_argument('--nireq', type=int, default=4,
                        help='Number of inference requests (default: 4)')
    parser.add_argument('--config', default=None,
                        help='Path to config file (default: reads from config.json)')

    args = parser.parse_args()

    # Load config
    sqlite_db_path = None
    clear_on_run = True
    clear_outputs = True
    use_case_id = 'pipeline_defects_detection'
    schema = None
    detection_mapping = None
    task = 'detect'
    modality = 'image'
    class_names = None
    sensor_model_path = None
    sensor_data_path = None
    handler_name = None
    fusion_weights = None
    config = {}

    if args.config:
        config_path = args.config
        config_parts = Path(config_path).parts
        if len(config_parts) >= 3 and config_parts[-1] == 'config.yaml':
            use_case_id = config_parts[-2]
    else:
        # Read default use case from config.json
        with open('config.json', 'r') as f:
            main_config = json.load(f)
        use_case_id = main_config.get('default-use-case', 'pipeline_defects_detection')
        config_path = f'config/{use_case_id}/config.yaml'

    if Path(config_path).exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Load task and modality
            inference_cfg = config.get('inference', {})
            task = inference_cfg.get('task', 'detect')
            handler_name = inference_cfg.get('handler')
            modality = config.get('modality', 'image')

            # Load class names
            names = config.get('names', {})
            if names:
                class_names = names

            # Load sensor config (for sensor/multi modality)
            sensor_cfg = config.get('sensor', {})
            sensor_model_path = sensor_cfg.get('model_path')
            sensor_data_path = sensor_cfg.get('data_path')
            fusion_weights = config.get('fusion_weights')

            # Load SQLite config
            sqlite_cfg = config.get('sqlite', {})
            sqlite_db_path = sqlite_cfg.get('db_path', f'out/{use_case_id}/sql_data/detections.db')
            clear_on_run = sqlite_cfg.get('clear_on_run', True)
            clear_outputs = sqlite_cfg.get('clear_outputs', True)

            # Load schema and detection_mapping from config
            schema = config.get('schema')
            if schema:
                # Inject class names into schema for get_schema() LLM context
                if names:
                    schema['class_names'] = names

            detection_mapping = config.get('detection_mapping')

            # Load model path from config if not provided via --model
            if not args.model:
                args.model = inference_cfg.get('model_path', f'models/ov_models/{use_case_id}/image/best.xml')
            
            # Load images path from config if user didn't override via --images
            config_images_path = inference_cfg.get('images_path')
            if not args.images and config_images_path:
                args.images = config_images_path
            
            # Load the configured image limit unless explicitly overridden.
            if args.num_images is None:
                args.num_images = inference_cfg.get('num_images')

            # Load video path from config if not provided via --video
            if not args.video:
                input_mode = inference_cfg.get('input_mode', 'images')
                if input_mode == 'video':
                    args.video = inference_cfg.get('video_path')
        except Exception as e:
            print(f"⚠️  Could not load config: {e}")
            print("   Continuing with JSONL output only...")

    # Fallback detection_mapping if config doesn't provide one
    if not detection_mapping:
        detection_mapping = {
            'frame_id': '__frame_index__',
            'label': 'detection.label',
            'confidence': 'detection.confidence',
            'x': 'scale_w:detection.bounding_box.x_min',
            'y': 'scale_h:detection.bounding_box.y_min',
            'width': 'scale_w_diff:detection.bounding_box.x_max-detection.bounding_box.x_min',
            'height': 'scale_h_diff:detection.bounding_box.y_max-detection.bounding_box.y_min',
        }

    # Final fallback if model still not set
    if not args.model:
        args.model = f'models/ov_models/{use_case_id}/image/best.xml'

    # Final fallback if images path still not set
    if not args.images:
        args.images = f'datasets/{use_case_id}/images/val'

    # Default output path: out/<use_case_id>/detections.jsonl
    out_dir = f'out/{use_case_id}'
    if not args.output:
        args.output = f'{out_dir}/detections.jsonl'

    # Default sqlite path if not set from config
    if not sqlite_db_path:
        sqlite_db_path = f'{out_dir}/sql_data/detections.db'

    # Default model_proc path (alongside model XML)
    if not args.model_proc:
        args.model_proc = str(Path(args.model).parent / 'model_proc.json')

    # Clean outputs directory if specified in config
    if clear_outputs:
        clean_outputs(out_dir=out_dir)
        print()
    else:
        print(f"📌 Keeping existing outputs (clear_outputs: false)\n")
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Use the configured device unless explicitly overridden by --device.
    if args.device is None:
        args.device = config.get('inference', {}).get('device', 'GPU')

    run_inference(args.model, args.model_proc, args.images, args.output,
                  args.device, args.conf_threshold, args.num_images,
                  args.nireq, sqlite_db_path, clear_on_run, args.video,
                  args.inference_interval, schema=schema,
                  detection_mapping=detection_mapping, out_subdir=use_case_id,
                  task=task, modality=modality, class_names=class_names,
                  sensor_model_path=sensor_model_path,
                  sensor_data_path=sensor_data_path,
                  fusion_weights=fusion_weights,
                  handler_name=handler_name,
                  config=config)
