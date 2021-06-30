import json
import os
from flask import Flask, request, Response
from src.utils.model import ModelSetup, ModelUtil
from src.utils.preprocess import ImageProcessing

app = Flask(__name__)

MODEL_PATH = ModelSetup(os.path.join(
    os.getcwd(), 'model')).get_latest_version()
model = ModelUtil(MODEL_PATH)
CLASS_LABELS = ['Cats', 'Dogs']
INPUT_RESIZE = model.model.input_shape[1:3]


@app.get("/")
def home():
    return "Server is up and running.", 200


@app.route("/classify/withImageFile", methods=["POST"])
def classify_with_image_file():
    try:
        file_obj = request.files['image']
        data = ImageProcessing(file_obj, type='fileobj').convert_to_model_input(
            resize=INPUT_RESIZE)
        result = model.predict(data)

        res_obj = {}
        res_obj['message'] = "success"
        res_obj['result'] = dict(zip(CLASS_LABELS, map(str, result[0])))
        return Response(json.dumps(res_obj), status=200, mimetype='application/json')
    except Exception as e:
        res_obj = {}
        res_obj['message'] = "failure"
        res_obj['error'] = str(e)
        return Response(json.dumps(res_obj), status=400, mimetype='application/json')


@app.route("/classify/withImageString", methods=["POST"])
def classify_with_image_string():
    try:
        image_obj = json.loads(request.data)
        image_str = image_obj['imageString'].split(',')[1]
        data = ImageProcessing(image_str, type='fileobj').convert_to_model_input(
            resize=INPUT_RESIZE)
        result = model.predict(data)

        res_obj = {}
        res_obj['message'] = "success"
        res_obj['result'] = dict(zip(CLASS_LABELS, map(str, result[0])))
        return Response(json.dumps(res_obj), status=200, mimetype='application/json')
    except Exception as e:
        res_obj = {}
        res_obj['message'] = "failure"
        res_obj['error'] = str(e)
        return Response(json.dumps(res_obj), status=400, mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
