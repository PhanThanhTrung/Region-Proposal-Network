import cv2
import numpy as np
from model import RPN
from model import losses
from keras.optimizers import Adam
from data_processing import data_processing
import time
model=RPN.model()
optimizer=Adam(lr=1e-5)
model.compile(optimizer=optimizer, loss=[losses.regression_loss, losses.classification_loss])
#model.fit_generator(data_processing.batch_generator(), steps_per_epoch= 1024, epochs=20)
batch_generator= data_processing.batch_generator()
for i in range(20):
    print("Epoch ",i,": ")
    for j in range(1024):
        image, [y_truth_cls, y_truth_reg]= next(batch_generator)
        output=model.predict([image])
        print("image shape: ", image.shape, "   y_pred_reg shape: ", output[0].shape, "  y_truth_reg shape: ", y_truth_reg.shape, "    y_pred_cls shape: ", output[1].shape, "    y_truth_cls shape: ", y_truth_cls.shape) 
        time.sleep(30)
