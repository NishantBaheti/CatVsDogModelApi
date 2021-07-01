"""
app contains

routes
global configs
logger

"""
import json
import os
from logging.config import dictConfig

from flask import Flask, request, Response
from flask.logging import create_logger
from flask_cors import cross_origin

from src.utils.ml_model import ModelSetup, ModelUtil
from src.utils.preprocess import ImageProcessing

if 'logs' not in os.listdir():
    os.mkdir('logs')

ENV = os.environ.get('FLASK_ENV') or 'PRODUCTION'
LOG_LEVEL_MAP = {
    "DEVELOPMENT": "DEBUG",
    "PRODUCTION": "INFO"
}

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '%(asctime)s || [%(levelname)s] [%(module)s:%(lineno)d] :: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }, 'file': {
        'class': 'logging.handlers.RotatingFileHandler',
        'filename': './logs/api.log',
        'mode': 'a',
        'formatter': 'default'
    }},
    'root': {
        'level': LOG_LEVEL_MAP.get(ENV.upper()) or LOG_LEVEL_MAP.get('PRODUCTION'),
        'handlers': ['wsgi', 'file']
    }
})


app = Flask(__name__)

_logger = create_logger(app)

_logger.info(
    "*-------------------- API starting in %s--------------------*", ENV)

# get model path with latest version
MODEL_PATH = ModelSetup(
    os.path.join(os.getcwd(), 'model')
).get_latest_version()
_logger.info("model path found : %s", MODEL_PATH)

# classification labels
CLASS_LABELS = ['Cat', 'Dog']

# load model
try:
    model = ModelUtil(MODEL_PATH)
except Exception as error:
    _logger.error("Error in loading model")
    _logger.error(str(error))
    raise error
_logger.info("ML model loaded successfully.")


# input shape of image for model
INPUT_RESIZE = model.input_shape[1:3] if model.input_shape is not None else (
    128, 128)
_logger.info("input resize parameter %s", INPUT_RESIZE)


############################################################################
########################### Routes #########################################

############################################################################
# home
@app.get("/")
def home() -> Response:
    """home route.

    Returns:
        Response: response.
    """
    res_obj = {
        "result": "success",
        "message": "Server is up and running."
    }
    return Response(json.dumps(res_obj), status=200, mimetype='appllcation/json')


############################################################################
# /classify/withImageFile

@app.route("/classify/withImageFile", methods=["POST"])
@cross_origin()
def classify_with_image_file() -> Response:
    """Classification with Image file sent with request.

    Returns:
        Response: response.
    """
    try:
        file_obj = request.files['image']
        data = ImageProcessing(file_obj).convert_to_model_input(
            resize=INPUT_RESIZE)
        result = model.predict(data)

        res_obj = {
            "result": "success",
            "message": "classification process successful.",
            "mlOutput": dict(zip(CLASS_LABELS, map(str, result[0])))
        }
        return Response(json.dumps(res_obj), status=200, mimetype='application/json')
    except Exception as error:
        _logger.error("%s", error)
        res_obj = {
            'result': 'failure',
            'message': str(error)
        }
        return Response(json.dumps(res_obj), status=400, mimetype='application/json')


############################################################################
# /classify/withImageString

@app.route("/classify/withImageString", methods=["POST"])
@cross_origin()
def classify_with_image_string() -> Response:
    """Classification with Image string/stream.

    Returns:
        Response: response.
    """
    try:
        file_obj = json.loads(request.data)

        if 'imageString' in file_obj.keys():
            image_str = file_obj['imageString'].split(',')[1]
            data = ImageProcessing(image_str).convert_to_model_input(
                resize=INPUT_RESIZE)
            result = model.predict(data)

            res_obj = {
                'result': 'success',
                'message': 'classification result successful.',
                'mlOutput': dict(zip(CLASS_LABELS, map(str, result[0])))
            }
            return Response(json.dumps(res_obj), status=200, mimetype='application/json')

        _logger.error('imageString key is missing')
        res_obj = {
            'result': 'failure',
            'message': 'imageString key is missing. please check input parameters again.'
        }
        return Response(json.dumps(res_obj), status=400, mimetype='application/json')

    except Exception as error:
        _logger.error("%s", error)
        res_obj = {
            'result': 'failure',
            'message': str(error)
        }
        return Response(json.dumps(res_obj), status=400, mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
