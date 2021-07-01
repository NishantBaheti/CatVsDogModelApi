import tensorflow as tf
from PIL import Image
import io
import numpy as np

model_path = '../modelapi/model/model-v0.0.1'
dataset_path = "../dataset/cat/cat.1.jpg"

model = tf.keras.models.load_model(model_path)
resize_param = model.input_shape[1:3]

with open(dataset_path, "rb") as file:
    data = file.read()
    img = Image.open(io.BytesIO(data))

img = img.resize(size=resize_param)
# img.show()
print(type(img))


img = np.array(img)
data = np.expand_dims(img, axis=0)
# print(data)

result = model.predict(x=data)
print(result)
