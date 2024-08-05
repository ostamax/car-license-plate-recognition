import os
from PIL import Image
from flask import Flask, render_template, request

from src.car_plate_detector import CarPlateDetector
from src.car_plate_ocr import CarPlateOCR
from src.utils import Utils

from config.config import *

app = Flask(__name__)


cpd = CarPlateDetector()
cpocr = CarPlateOCR()
utils = Utils()
cpd.warm_up_inference()

@app.route('/', methods=['GET'])
def start_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = os.path.join(INPUT_DATA, imagefile.filename)

    imagefile.save(image_path)
    img = Image.open(imagefile)
    utils.resize_and_save_image(img, os.path.join(STATIC_DATA, RAW_IMAGE_NAME))

    cropped_plates, bboxes_coords = cpd.run_detect(image_path)
    recognized_text, idx = list(map(cpocr.run_ocr, cropped_plates))[0]
    utils.crop_and_save_image(img, bboxes_coords[idx], os.path.join(STATIC_DATA, DETECTED_PLATE_NAME))

    return render_template('index.html', prediction=recognized_text)


if __name__ == '__main__':
    app.run(port=3000, debug=True)