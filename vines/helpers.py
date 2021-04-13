import numpy as np
def my_round(n):
    delta = n - np.floor(n)
    if delta < 0.5:
        part = np.floor(n)
    else:
        part = np.ceil(n)
    return part