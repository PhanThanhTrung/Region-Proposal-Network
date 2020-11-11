import cv2
import numpy as np
from model import RPN
from model import losses
from keras.optimizers import Adam
path='./test_image.JPEG'
image=cv2.imread(path)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input=np.array([image])
model=RPN.RPN()
optimizer=Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss=[losses.classification_loss, losses.regression_loss])
output=model.predict(input)
print(output[0].shape)