import easyocr
import numpy as np

class CarPlateOCR:
    """Class that implements the logic of OCR on a car lisence plate."""

    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'])

    def find_biggest_bbox(self, ocr_results: list) -> int:
        """
        Method which finds the biggest detected text bounding box according to its area.

        Args:
            ocr_results (list): list of OCR model inference results

        Returns:
            int: index of the biggest bounding box
        """
        areas_list = []
        if len(ocr_results) > 1: 
            for res_tuple in ocr_results:
                area = (res_tuple[0][1][0] - res_tuple[0][0][0]) * (res_tuple[0][2][1] - res_tuple[0][2][0])
                areas_list.append(area)
            return areas_list.index(min(areas_list))
        else:
            return 0
        
    def filter_alphanumeric_chars(self, license_plate_text: str) -> str:
        """FIlters non-alphanumeric characters out.

        Args:
            license_plate_text (str): raw recognized text

        Returns:
            str: processed text
        """
        
        filtered_string = ''.join(letter for letter in license_plate_text if letter.isalnum())
        return filtered_string

    def run_ocr(self, image: np.array) -> tuple[str, int]:
        """
        Method that runs OCR model inference.

        Args:
            image (np.array): input image of cropped license plate

        Returns:
            tuple[str, int]: [processed text, index of the largest bounding box]
        """
        result = self.reader.readtext(image)

        if len(result) > 0:
            idx = self.find_biggest_bbox(ocr_results=result)
            result = result[idx]
            filtered_result = self.filter_alphanumeric_chars(result[1].upper())
            return filtered_result, idx
        else:
            return "", None