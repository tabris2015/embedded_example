import cv2
import numpy as np

from app.models import Detection

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(image, detection: Detection) -> np.ndarray:
    for box, label, confidence in zip(
        detection.boxes, detection.labels, detection.confidences
    ):
        # Draw bounding_box
        start_point = box[0], box[1]
        end_point = box[2], box[3]
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        result_text = label + " (" + str(confidence) + ")"
        text_location = (MARGIN + box[0], MARGIN + ROW_SIZE + box[1])
        cv2.putText(
            image,
            result_text,
            text_location,
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    return image
