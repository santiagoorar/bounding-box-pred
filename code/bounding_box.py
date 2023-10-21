import numpy as np
from typing import Tuple
import PIL

def calculate_bounding_box(x: PIL.Image.Image) -> Tuple[float, float, float, float]:
    """
    The idea is to create a function to calculate the real bounding box of the MNIST dataset.
    
    :param x: A 28x28 grayscale image from the MNIST dataset.
    :return: Tuple (left, right, up, down).
    """
    
    bounding_box = None
    
    # the idea is to find the first pixel that is not zero (not black color) from the left, right, up, down:
    left = np.argmax(np.array(x).sum(axis=0) != 0)  # to indicate "left", sum(axis=0) is used. Argmax to find the first True value (first non black pixel)
    right = len(np.array(x).sum(axis=0)) - np.argmax(np.flip(np.array(x).sum(axis=0)) != 0)  # same idea as left, but now
    up = np.argmax(np.array(x).sum(axis=1) != 0)
    down = len(np.array(x).sum(axis=0)) - np.argmax(np.flip(np.array(x).sum(axis=1)) != 0)
    
    bounding_box = (float(left), float(right), float(up), float(down))  
    
    return bounding_box