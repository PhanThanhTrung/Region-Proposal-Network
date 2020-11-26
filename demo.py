import cv2
import numpy as np
from model import RPN
from model import losses
from keras.optimizers import Adam
from data_processing import data_processing
import time
model=RPN.model()
print(model.summary())
optimizer=Adam(lr=1e-4)
model.compile(optimizer=optimizer, loss=[losses.regression_loss, losses.classification_loss])
model.fit_generator(data_processing.batch_generator(), steps_per_epoch= 1024, epochs=20)