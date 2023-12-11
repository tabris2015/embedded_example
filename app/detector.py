from app.models import Detection, PredictionType
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.detections import DetectionResult

MAX_DETECTIONS = 10


class MediapipeStreamObjectDetector:
    def __init__(self, model, threshold=0.5):
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            max_results=MAX_DETECTIONS,
            result_callback=self.detection_callback,
            score_threshold=threshold,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.result_list = []

    def detection_callback(
        self,
        result: DetectionResult,
        output_image: mp.Image,
        timestamp_ms: mp.Timestamp,
    ):
        if isinstance(timestamp_ms, mp.Timestamp):
            timestamp_ms = timestamp_ms.value
        boxes = []
        labels = []
        confidences = []
        for detection in result.detections:
            bbox = detection.bounding_box
            boxes.append(
                [
                    bbox.origin_x,
                    bbox.origin_y,
                    bbox.origin_x + bbox.width,
                    bbox.origin_y + bbox.height,
                ]
            )
            labels.append(detection.categories[0].category_name)
            confidences.append(detection.categories[0].score)
        detection = Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels,
            confidences=confidences,
            timestamp_ms=timestamp_ms,
        )
        self.result_list.append(detection)
