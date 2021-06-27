import tensorflow as tf
from PIL import Image
import io
import numpy as np

model_path = '../model/model-v0.0.1'

model = tf.keras.models.load_model(model_path)

with open("../dataset/cat/cat.1.jpg", "rb") as file:
    data = file.read()
    img = Image.open(io.BytesIO(data))

img = img.resize(size=(128, 128))
# img.show()


img = np.array(img)
data = np.expand_dims(img, axis=0)
# print(data)

print(model)
print(dir(model))

result = model.predict(x=data)
print(result)
