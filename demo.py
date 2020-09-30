import cv2
import numpy as np
from RPN import *
path='./test_image.JPEG'
image=cv2.imread(path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input=np.array([image])
model=model()
output=model.predict(input)
print(output[0].shape)