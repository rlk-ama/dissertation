import sys
import numpy as np


def underflow(density):
    def wrapper(*args, **kwargs):
        try:
            density(*args, **kwargs)
        except FloatingPointError:
            return sys.float_info[3]
        else:
            return density(*args, **kwargs)
    return wrapper
