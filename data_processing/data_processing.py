import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import random
import numpy as np
import config
from utils import data_utils, model_utils
import xml.etree.ElementTree as ET

def parse_xml(file_name):
    details={}
    details['image_path']=config.train_image_folder+file_name.split('/')[-1][:-3]+'jpg'
    details['bounding_box']=[]
    root= ET.parse(file_name).getroot()
    for bbox in root.findall("object/bndbox"):
        x_max =int(bbox.find("xmax").text)
        y_max=int(bbox.find("ymax").text)
        x_min=int(bbox.find("xmin").text)
        y_min=int(bbox.find("ymin").text)

        details['bounding_box'].append([x_min, y_min, x_max, y_max])
    return details

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
    featuremap_height, featuremap_width, nb_anchor, _ = all_anchors.shape
    dense_iou = model_utils.calculate_iou(all_anchors, bboxes)
    y_truth_reg = model_utils.transform_box(all_anchors, bboxes, dense_iou)
    anchors_label = model_utils.anchor_matching(
        dense_iou,
        higher_threshold=config.higher_threshold,
        lower_threshold=config.lower_threshold)
    anchors_label = model_utils.refined_anchors(anchors_label, all_anchors,
                                                image)
    #because there are too much posivitve and negative anchors, we randomly select 256 of them with the ratio of pos/neg=1/2
    nb_foreground = int(config.batch_size / 3)
    nb_background = int(config.batch_size - nb_foreground)
    pos_anchor = np.where(anchors_label == 1)[0]
    neg_anchor = np.where(anchors_label == 0)[0]
    foreground_choice = random.sample(list(pos_anchor), nb_foreground)
    background_choice = random.sample(list(neg_anchor), nb_background)
    anchors_label[pos_anchor] = -1
    anchors_label[foreground_choice] = 1
    anchors_label[neg_anchor] = -1
    anchors_label[background_choice] = 0

    y_truth_cls = anchors_label.reshape((featuremap_height,
                                                 featuremap_width, nb_anchor,
                                                 1))
    y_truth_reg= np.concatenate((y_truth_reg, y_truth_cls), axis=-1)
    return image, y_truth_cls, y_truth_reg

def batch_generator():
    all_file=os.listdir(config.train_anno_file_path)
    all_data=all_file.copy()
    while True:
        random.shuffle(all_data)
        for sample in all_data:
            details=parse_xml(config.train_anno_file_path+sample)
            # try:
            #     image, y_truth_cls, y_truth_reg= produce_batch(details)
            #     image=np.expand_dims(image, axis=0)
            #     y_truth_cls=np.expand_dims(y_truth_cls, axis=0)
            #     y_truth_reg=np.expand_dims(y_truth_reg,axis=0)

            #     yield image, [y_truth_cls, y_truth_reg]
            # except Exception as e:
            #     print(e)
            #     break
            image, y_truth_cls, y_truth_reg= produce_batch(details)
            image=np.expand_dims(image, axis=0)
            y_truth_cls=np.expand_dims(y_truth_cls, axis=0)
            y_truth_reg=np.expand_dims(y_truth_reg,axis=0)
            y_truth_cls=np.array(y_truth_cls, dtype='float32')
            y_truth_reg=np.array(y_truth_reg, dtype='float32')

            yield image, [y_truth_cls, y_truth_reg]