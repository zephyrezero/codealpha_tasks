import numpy as np
from ultralytics import YOLO
import cv2
import argparse
import supervision as sv
import torch

torch.cuda.set_device(0)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="yololive")
    parser.add_argument('--weres', default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def initialize_model(model_path: str) -> YOLO:
    return YOLO(model_path)

def initialize_video_capture(frame_width: int, frame_height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return cap

def process_frame(model: YOLO, frame: np.ndarray, box_annotator: sv.BoxAnnotator) -> np.ndarray:
    results = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(results)
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    return annotated_frame

def display_frame(frame: np.ndarray, window_name: str = 'yolov8l_real_time_det') -> bool:
    cv2.imshow(window_name, frame)
    if cv2.waitKey(30) == 27:  # ESC key
        return False
    return True

def main():
    args = parse_arguments()
    frame_width, frame_height = args.weres

    cap = initialize_video_capture(frame_width, frame_height)
    model = initialize_model('yolov8l.pt')
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = process_frame(model, frame, box_annotator)
        if not display_frame(annotated_frame):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
