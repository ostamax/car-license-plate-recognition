from PIL import Image

class Utils:
    """Class that implement all compulsory methods for image processing"""

    def __init__(self) -> None:
        pass

    def resize_and_save_image(self, image: Image, save_path: str) -> None:
        """
        Resizes and saves image to be printed in future on a web-page.

        Args:
            image (Image): input image
            save_path (str): path for image to be saved in
        """
        width, height = image.size
        if height > 300:
            hratio = 300 / height
            wsize = int(width * hratio)
            image = image.resize((wsize, 300), Image.Resampling.LANCZOS)
        
        image.save(save_path)

    def crop_and_save_image(self, image: Image, bbox_coords: list, save_path: str) -> None:
        """
        Crops detected lisence plate and saves it for furure usage.

        Args:
            image (Image): input image
            bbox_coords (list): list of license plate bounding box coordinates
            save_path (str): path for image to be saved in
        """
        image = image.crop(tuple(int(coord) for coord in bbox_coords))
        image.save(save_path)

