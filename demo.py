import cv2
import numpy as np
from model import RPN
from model import losses
from keras.optimizers import Adam
from data_processing import data_processing

model=RPN.model()
optimizer=Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss=[losses.classification_loss, losses.regression_loss])
model.fit_generator(data_processing.batch_generator(), epochs=20)
 