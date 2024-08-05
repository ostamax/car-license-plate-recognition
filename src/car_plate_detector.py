import os
import cv2
import numpy as np
from ultralytics import YOLO

from config.config import *

class CarPlateDetector:
    """Class that implements the logic of a car lisence plate detection."""

    def __init__(self) -> None:
        self.weigths = os.path.join(WEIGHTS_PATH, LICENCE_PLATE_WEIGHTS_FILE)
        self.conf = CONFIDENCE_THRESHOLD
        self.iou = IOU_THRESHOLD
        self.imgsz = IMGSZ

        self.model = YOLO(self.weigths)

    def warm_up_inference(self) -> None:
        """Method that implements warming up inference"""
        warm_up_array = np.zeros((640, 640, 3))
        self.model.predict(warm_up_array, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
    
    @staticmethod
    def process_results(bounding_boxes: list, image_path: str) -> list:
        """
        Processes detection results and crops detected license plate.

        Args:
            bounding_boxes (list): list of detected objects' coordinates
            image_path (str): image path

        Returns:
            list: list of cropped images
        """

        cropped_images = []
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for box in bounding_boxes:
            cropped_plate = image[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            cropped_images.append(cropped_plate)
        
        return cropped_images

    def run_detect(self, image: str) -> tuple[list, list]:
        """
        Method that implements detection model inference and further results processing.

        Args:
            image (str): image path

        Returns:
            tuple[list, list]: [cropped images of detected license plates,
                                bounding boxes of detected objects]
        """

        results = self.model.predict(image, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        bounding_boxes = results[0].boxes.xyxy.tolist()
        detected_plates = self.process_results(bounding_boxes, image)
        
        return detected_plates, bounding_boxes
