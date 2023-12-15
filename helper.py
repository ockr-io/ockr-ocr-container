import base64
import numpy as np
import cv2


def decode_base64(data):
    numpy_array = np.fromstring(base64.b64decode(data), np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    return image
