"""
Machine Learning Model management module
"""
import os
import re
from typing import List, Union
import logging

import numpy as np
import tensorflow as tf

logging.getLogger(__name__)

class ModelSetup:
    """Setup Model with the path of directory

    Args:
        dir_path (str): directory path for models.

    Raises:
        AttributeError: If directory doesn't exists.
    """

    def __init__(self, dir_path: str):
        """Constructor
        """
        if os.path.isdir(dir_path):
            self.dir_path = str(os.path.abspath(dir_path))
            self._models = [name for name in os.listdir(
                self.dir_path) if name.startswith("model")]
        else:
            raise AttributeError(f"{dir_path} doesn't exist.")

    @property
    def models(self) -> List[str]:
        """get model paths.

        Returns:
            List[str]: list of model paths.
        """
        return [os.path.join(self.dir_path, name) for name in self._models]

    @classmethod
    def get_version_from_string(cls, in_str: str) -> str:
        """get version from a string using regex.

        Args:
            in_str (str): input string.

        Returns:
            str: version string.
        """
        regex_str = r'\v?(\d+)\.(\d+)\.(\d+)'
        match = re.search(regex_str, in_str)
        if match:
            return match.group()
        return '0.0.0'

    def get_latest_version(self) -> str:
        """Get latest verion model path.

        Returns:
            str: model path.
        """
        latest_version = sorted(
            self.models, key=self.get_version_from_string, reverse=True)[0]
        return os.path.join(self.dir_path, latest_version)

    def get_specific_version(self, version: str) -> Union[str, None]:
        """Get specific versions model.

        Args:
            version (str): version string. eg.- '1.0.0','v1.0.0'.

        Returns:
            Union[str,None]: model path or None if not found.
        """
        version = self.get_version_from_string(version)
        for model in self._models:
            if version == self.get_version_from_string(model):
                return os.path.join(self.dir_path, model)
        return None

    def get_by_name(self, name: str) -> Union[str, None]:
        """Get by name of the model.

        Args:
            name (str): model name.

        Returns:
            Union[str, None]: model path or None if not found.
        """
        if name in self._models:
            return os.path.join(self.dir_path, name)
        return None


class ModelUtil:
    """Machine Learning Model Utility.

    Args:
        model_path (str): machine learning model path.
    """

    def __init__(self, model_path: str):
        """Constructor"""
        self._model = tf.keras.models.load_model(model_path)

    @property
    def model(self):
        """property to get model"""
        return self._model

    @property
    def input_shape(self):
        """property to get input matrix shape"""
        if self._model is not None:
            return self._model.variables[0].shape
        return None

    @staticmethod
    def _add_dim_for_model(mat: np.ndarray) -> np.ndarray:
        """Add dimension for machine learning model input.

        Args:
            mat (np.ndarray): input matrix.

        Returns:
            np.ndarray: output matrix.
        """
        mat = np.expand_dims(mat, axis=0)  # adding extra dimension
        return mat

    def predict(self, image_mat: np.ndarray) -> np.ndarray:
        """Prediction method.

        Args:
            image_mat (np.ndarray): input image matrix.

        Returns:
            np.ndarray: output prediction.
        """
        if len(image_mat.shape) == 3:
            # it was one image matrix. converting it into a row
            image_mat = self._add_dim_for_model(image_mat)

        result = self._model.predict(x=image_mat)
        return result


# if __name__ == "__main__":
#     model_setup = ModelSetup("../../model")

#     print(model_setup.models)

#     model_path = model_setup.get_latest_version()

#     print(model_setup.get_latest_version())

#     print(model_setup.get_specific_version('0.0.1'))

#     print(model_setup.get_by_name('model-v0.0.1'))

#     model = ModelUtil(model_path)

#     from PIL import Image
#     import io

#     with open("../../dataset/cat/cat.1.jpg", "rb") as file:
#         data = file.read()
#         img = Image.open(io.BytesIO(data))

#     img = img.resize(size=(128, 128))
#     img = np.array(img)
#     result = model.predict(img)
#     print(result)
