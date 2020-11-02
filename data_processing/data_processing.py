import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import config
from utils import data_utils, model_utils


def load_data():
    """
        load images and annotation to array
    """
    all_image = []
    with open(config.train_anno_file_path, "r") as f:
        while True:
            image = {}
            line = f.readline()
            if line == "":
                break

            if ".jpg" in line:
                #print(line)
                image['image_path'] = (config.train_image_folder +
                                       line).strip(" ").strip("\n")
            next_line = f.readline()
            number_of_object = int(next_line)
            if number_of_object != 0:
                image['bounding_box'] = []
                for i in range(number_of_object):
                    next_line = f.readline()
                    bbox = list(
                        map(int,
                            next_line.strip("\n").strip(" ").split(" ")))[:4]
                    image['bounding_box'].append(bbox)
            else:
                next_line = f.readline()
            all_image.append(image)
    return all_image


def visualize(all_image):
    for image in all_image:
        print(image['image_path'])
        img = cv2.imread(image['image_path'])
        for bbox in image.get('bounding_box'):
            left, bottom = bbox[0], bbox[1]
            right, top = left + bbox[2], bottom + bbox[3]
            img = cv2.rectangle(img, (left, top), (right, bottom),
                                color=(0, 255, 0),
                                thickness=2)
        cnt = 0
        cv2.imshow('frame', img)
        cv2.waitKey(1)


def produce_batch(image_detail):
    image = cv2.imread(image_detail.get('image_path'))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = image_detail.get('bounding_box')
    bboxes = np.array(bboxes)
    bboxes = data_utils.normalize_box(bboxes)

    all_anchors = model_utils.anchors_generator(image, config.output_stride,
                                                config.scales, config.ratios)
    dense_iou = model_utils.calculate_iou(all_anchors, bboxes)
    anchors_label = model_utils.anchor_matching(
        dense_iou,
        higher_threshold=config.higher_threshold,
        lower_threshold=config.lower_threshold)
    anchors_label = model_utils.refined_anchors(anchors_label, all_anchors,
                                                image)

    anchors_label = anchors_label.reshape(
        shape=(anchors_label.shape[0] * anchors_label.shape[1] *
               anchors_label.shape[2], 1))
    
    nb_foreground=config.batch_size/3
    nb_background=config.batch_size-nb_foreground
    foreground_choice=np.random.choice(np.array(np.where(anchors_label==1)).T,size=nb_foreground)

anchors_label=np.random.randint(-1,2, size=14*14*9*1).reshape((14*14*9,1))
nb_foreground=config.batch_size/3
nb_background=config.batch_size-nb_foreground
#foreground_choice=np.random.choice(np.array(np.where(anchors_label==1)).T,size=nb_foreground)
print(anchors_label.shape)
print(np.array(np.where(anchors_label==1)).shape)
print(anchors_label[np.where(anchors_label==1)[0]])