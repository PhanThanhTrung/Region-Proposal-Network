import numpy as np

def normalize_box(bbox):
    left, bottom, width, height = bbox[:,0],bbox[:,1],bbox[:,2],bbox[:,3]
    center_x, center_y = left + width // 2, bottom + height // 2
    return np.array([center_x, center_y, width, height]).T

