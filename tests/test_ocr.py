import os
import cv2
import unittest

from src.car_plate_ocr import CarPlateOCR

class TestOCRModule(unittest.TestCase):

    def setUp(self) -> None:
        self.car_ocr = CarPlateOCR()
        self.image_to_test_on_1 = cv2.imread(os.path.join("data", "1.jpg"))
        self.image_to_test_on_2 = cv2.imread(os.path.join("data", "2.jpg"))
        self.bboxes_to_compare_areas = [([[94, 22], [493, 22], [493, 120], [94, 120]], 'test', 0.25), ([[383, 117], [499, 117], [499, 137], [383, 137]], 'text', 0.99)]

    def test_run_ocr_non_empty(self):
        
        result = self.car_ocr.run_ocr(self.image_to_test_on_1)
        self.assertEqual(result, ("AB44887", 0))

    def test_run_ocr_on_empty(self):
        
        result = self.car_ocr.run_ocr(self.image_to_test_on_2)
        self.assertEqual(result, ("", None))

    def test_find_biggest_bbox(self):
        
        result = self.car_ocr.find_biggest_bbox(self.bboxes_to_compare_areas)
        self.assertEqual(result, 0)

    





