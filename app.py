import os
import json
from flask import Flask, Blueprint, request, Response
from src.model import ModelSetup, ModelUtil
from src.preprocess import ImageProcessing

app = Flask(__name__)

MODEL_PATH = ModelSetup(os.path.join(os.getcwd(), 'model')).get_latest_version()
model = ModelUtil(MODEL_PATH)
CLASS_LABELS = ['Cats','Dogs']
INPUT_RESIZE = (128,128)


@app.get("/")
def home():
    return "Server is up and running.",200


@app.route("/classify/withImageFile", methods=["POST"])
def classify_with_image_file():
    file_obj = request.files['image']
    data = ImageProcessing(file_obj, type='fileobj').convert_to_model_input(
        resize=INPUT_RESIZE)
    result = model.predict(data)

    res_obj = {}
    res_obj['message'] = "success"
    res_obj['result'] = dict(zip(CLASS_LABELS,map(str,result[0])))
    return Response(json.dumps(res_obj), status=200, mimetype='application/json')

@app.route("/classify/withImageString", methods=["POST"])
def classify_with_image_string():
    image_obj = json.loads(request.data)
    image_str = image_obj['imageString'].split(',')[1]
    data = ImageProcessing(image_str, type='fileobj').convert_to_model_input(
        resize=INPUT_RESIZE)
    result = model.predict(data)

    res_obj = {}
    res_obj['message'] = "success"
    res_obj['result'] = dict(zip(CLASS_LABELS, map(str, result[0])))
    return Response(json.dumps(res_obj), status=200, mimetype='application/json')
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)
