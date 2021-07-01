"""
Preprocessing module.

"""

import base64
import io
import logging
from typing import Union

import numpy as np
from PIL import Image
from werkzeug.datastructures import FileStorage

logging.getLogger(__name__)

class ImageProcessing:
    """Image Processing class.

    Args:
        image_obj (Union[FileStorage, str]): Image object.

    Raises:
        ValueError: Type not accepted.
    """

    def __init__(self, image_obj: Union[FileStorage, str]):
        """Constructor
        """
        if isinstance(image_obj, FileStorage):
            self.image_obj = image_obj ## it is already a BytesIO object
        elif isinstance(image_obj, str):
            self.image_obj = io.BytesIO(base64.b64decode(image_obj))
        else:
            raise ValueError("Type not accepted.")

    @staticmethod
    def _add_dimension(mat: np.ndarray) -> np.ndarray:
        """add dimension to the numpy array.

        Because model takes image array as input with (a,b,c,d) shape.
        if the matrix has only three dims then one dim is added with this method.

        Args:
            mat (np.ndarray): input matrix.

        Returns:
            np.ndarray: output matrix.
        """
        mat = np.expand_dims(mat, axis=0)
        return mat

    def convert_to_image(self, resize: Union[tuple, int] = None) -> Image.Image:
        """Convert to image from object.

        Args:
            resize (Union[tuple, int], optional): resize the image. Defaults to None.

        Raises:
            ValueError: resize parameter only accepts tuple, list or int.

        Returns:
            Image.Image: Image.
        """
        img = Image.open(self.image_obj)
        if resize is not None:
            if isinstance(resize, (tuple)):
                img = img.resize(size=resize)
            elif isinstance(resize, (int)):
                img = img.resize(size=(resize, resize))
            else:
                raise ValueError(
                    "resize parameter only accepts tuple, list or int.")
        return img

    def convert_to_numpy(self, resize: Union[tuple, int] = None) -> np.ndarray:
        """Convert to numpy array.

        Args:
            resize (Union[tuple, int], optional): resize image parameter. Defaults to None.

        Returns:
            np.ndarray: image array.
        """
        img = self.convert_to_image(resize=resize)
        return np.array(img)

    def convert_to_model_input(self, resize: Union[tuple, int] = None) -> np.ndarray:
        """Convert to model suitable input.

        Args:
            resize (Union[tuple, int], optional): resize image parameter. Defaults to None.

        Returns:
            np.ndarray: image array.
        """
        model_input = self._add_dimension(self.convert_to_numpy(resize=resize))
        return model_input
