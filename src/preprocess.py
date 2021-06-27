import io
import base64
from PIL import Image
import numpy as np
from typing import Union, Tuple


class ImageProcessing:

    def __init__(self,image_obj,type):
        if type == 'fileobj':
            self.image_obj = image_obj.stream
        elif type == 'filestr':
            self.image_obj = io.BytesIO(base64.b64decode(image_obj))
        else:
            raise ValueError("only 'fileobj', 'filestr' are accepted.")

    def _add_dimension(self, mat):
        mat = np.expand_dims(mat, axis=0)
        return mat
            
    def convert_to_image(self,resize = None):
        img = Image.open(self.image_obj)

        if resize is not None:
            if isinstance(resize,(tuple)):
                img = img.resize(size=resize)
            elif isinstance(resize,(int)):
                img = img.resize(size=(resize,resize))
            else:
                raise ValueError("resize parameter only accepts tuple,list or int")
        return img

    def convert_to_numpy(self,resize=None):
        img = self.convert_to_image(resize=resize)
        return np.array(img)

    def convert_to_model_input(self,resize=None):
        model_input = self._add_dimension(self.convert_to_numpy(resize=resize))
        return model_input
    
