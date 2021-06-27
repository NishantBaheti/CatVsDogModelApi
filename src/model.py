import numpy as np
import tensorflow as tf
import os 
import re

class ModelSetup:
    def __init__(self,dir_path):
        if os.path.isdir(dir_path):
            self.dir_path = str(os.path.abspath(dir_path))
            self._models = [name for name in os.listdir(self.dir_path) if name.startswith("model")]
        else:
            raise AttributeError(f"{dir_path} doesn't exist.")

    @property
    def models(self):
        return [os.path.join(self.dir_path,name) for name in self._models]  

    def get_model_version(cls,in_str):
        regex_str = '\\v?(\d+)\.(\d+)\.(\d+)'
        match = re.search(regex_str, in_str)
        if match:
            return match.group()
        return '0.0.0'

    def get_latest_version(self):
        latest_version = sorted(self.models,key=self.get_model_version,reverse=True)[0]
        return os.path.join(self.dir_path,latest_version)

    def get_specific_version(self,version):
        for model in self._models:
            if version == self.get_model_version(model):
                return os.path.join(self.dir_path,model)
        return None

    def get_by_name(self,name):
        if name in self._models:
            return os.path.join(self.dir_path,name)
        return None


class ModelUtil:

    def __init__(self, model_path):
        if os.path.isdir(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise AttributeError(f"{model_path} doesn't exist.")

    def _add_dim_for_model(self,mat):
        mat = np.expand_dims(mat, axis=0) # adding extra dimension
        return mat

    def predict(self, image_mat):
        
        if len(image_mat.shape) == 3:
            image_mat = self._add_dim_for_model(image_mat) # it was one image matrix. converting it into a row

        result = self.model.predict(x=image_mat)
        return result


if __name__ == "__main__":

    model_setup = ModelSetup("../../model")

    print(model_setup.models)

    model_path = model_setup.get_latest_version()

    print(model_setup.get_latest_version())

    print(model_setup.get_specific_version('0.0.1'))

    print(model_setup.get_by_name('model-v0.0.1'))

    model = ModelUtil(model_path)

    from PIL import Image
    import io

    with open("../../dataset/cat/cat.1.jpg", "rb") as file:
        data = file.read()
        img = Image.open(io.BytesIO(data))

    img = img.resize(size=(128, 128))
    img = np.array(img)
    result = model.predict(img)
    print(result)
