"""Built-in inference handlers."""

from .dlstreamer_detect import DLStreamerDetectHandler
from .openvino_audio_text_classify import OpenVINOAudioTextClassifyHandler
from .openvino_image_text_classify import OpenVINOImageTextClassifyHandler
from .openvino_classify import OpenVINOClassifyHandler
from .openvino_detect import OpenVINODetectHandler
from .sensor_flat import SensorFlatHandler

__all__ = [
    "DLStreamerDetectHandler",
    "OpenVINOAudioTextClassifyHandler",
    "OpenVINOImageTextClassifyHandler",
    "OpenVINOClassifyHandler",
    "OpenVINODetectHandler",
    "SensorFlatHandler"
]
