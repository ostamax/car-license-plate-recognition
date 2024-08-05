import os
import cv2
import unittest

from src.car_plate_detector import CarPlateDetector

class TestDetectionModule(unittest.TestCase):
    
    def setUp(self) -> None:
        self.detector = CarPlateDetector()
        self.image_to_test_on_1 = os.path.join("data", "1.jpg")
        self.image_to_test_on_2 = os.path.join("data", "2.jpg")
        self.bbox_coordinates = [[0.0, 0.7128238081932068, 142.1454315185547, 53.20188522338867]]
    
    def test_run_detect_on_image_with_plate(self):
        
        result = self.detector.run_detect(self.image_to_test_on_1)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1], self.bbox_coordinates)

    def test_run_detect_on_image_without_plate(self):
        
        result = self.detector.run_detect(self.image_to_test_on_2)
        self.assertEqual(result, ([], []))

    



