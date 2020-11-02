import numpy as np


def IOU(bbox1, bbox2):
    box1_x1, box1_y1, box1_x2, box1_y2 = bbox1[0] - bbox1[2] // 2, bbox1[
        0] + bbox1[2] // 2, bbox1[1] - bbox1[3] // 2, bbox1[1] + bbox1[3] // 2
    box2_x1, box2_y1, box2_x2, box2_y2 = bbox2[0] - bbox2[2] // 2, bbox2[
        0] + bbox2[2] // 2, bbox2[1] - bbox2[3] // 2, bbox2[1] + bbox2[3] // 2

    overlap_x = min(box1_x2, box2_x2) - max(box1_x1, box2_x1, 0)
    overlap_y = min(box1_y2, box2_y2) - max(box1_y1, box2_y1, 0)
    overlap = overlap_x * overlap_y

    union = bbox1[2] * bbox1[3] + bbox2[2] * bbox2[3] - overlap

    return overlap / union


def __anchors_size(output_stride, scales, ratios):
    cell = np.array([output_stride, output_stride]).reshape((2, 1))
    scale = np.array(scales).reshape(1, 3)
    ratio = np.array(ratios).reshape(3, 2)
    anchor_scale = (cell * scale).T
    all_anchor_size = np.array([
        anchor_scale * ratio[0], anchor_scale * ratio[1],
        anchor_scale * ratio[2]
    ])
    all_anchor_size = np.reshape(all_anchor_size, (9, 2))
    return all_anchor_size


def anchors_generator(image, output_stride, scales, ratios):
    image_height, image_width, _ = image.shape
    feature_map_width, feature_map_height = image_width // output_stride, image_height // output_stride

    feature_map_width_array = np.arange(0, feature_map_width)
    feature_map_height_array = np.arange(0, feature_map_height)
    feature_map_tiles = np.array(
        np.meshgrid(feature_map_height_array, feature_map_width_array))
    feature_map_tiles = feature_map_tiles.transpose((2, 1, 0)) * output_stride
    feature_map_tiles = np.expand_dims(feature_map_tiles, axis=2)
    feature_map_tiles = np.repeat(feature_map_tiles, repeats=9, axis=-2)

    all_anchors_at_cell = __anchors_size(output_stride, scales, ratios)
    all_anchors_size = np.zeros(shape=(feature_map_height, feature_map_width,
                                       9, 2))
    all_anchors_size = all_anchors_size + all_anchors_at_cell

    all_anchors = np.concatenate((feature_map_tiles, all_anchors_size),
                                 axis=-1)
    return all_anchors


def calculate_iou(all_anchors, groundtruth):
    all_anchors = np.reshape(all_anchors,
                             (all_anchors.shape[0] * all_anchors.shape[1] *
                              all_anchors.shape[2], 4))
    groundtruth = np.squeeze(np.array(groundtruth))
    dense_iou = np.zeros(shape=(all_anchors.shape[0], groundtruth.shape[0]))
    for row in range(dense_iou.shape[0]):
        for column in range(dense_iou.shape[1]):
            dense_iou[row, column] = IOU(all_anchors[row], groundtruth[column])

    return dense_iou


def anchor_matching(dense_iou, higher_threshold, lower_threshold):
    """
    According to the paper, we assign a positive label to two kinds of anchors: 
    (i) the anchor/anchors with the highest Intersection-over- Union (IoU) overlap with a ground-truth box.
    (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box. 
    We assign a negative label to a non-positive anchor if its IoU ratio is lower than 0.3 for all ground-truth 
    boxes.
    """
    label_anchors = np.full((dense_iou.shape[0], 1), -1, dtype=int)
    max_overlap_with_anchor = np.array(np.max(dense_iou, axis=0))
    max_overlap_arg_with_anchor = np.where(
        dense_iou == max_overlap_with_anchor)[0]
    label_anchors[max_overlap_arg_with_anchor] = 1

    max_overlap_with_gt = np.sum(dense_iou >= higher_threshold, axis=1, keepdims=True)
    label_anchors[max_overlap_with_gt != 0] = 1

    max_overlap_with_gt = np.sum(dense_iou <= lower_threshold, axis=1, keepdims=True)
    label_anchors[max_overlap_with_gt == dense_iou.shape[1]] = 0

    return label_anchors


def refined_anchors(label_anchors, all_anchors, image):
    image_height, image_width = image.shape[0], image.shape[1]
    outside_anchors = np.where(
        all_anchors[..., 0] + all_anchors[..., 2] // 2 > image_width)
    label_anchors[outside_anchors] = -1
    outside_anchors = np.where(
        all_anchors[..., 0] - all_anchors[..., 2] // 2 < 0)
    label_anchors[outside_anchors] = -1
    outside_anchors = np.where(
        all_anchors[..., 1] + all_anchors[..., 3] // 2 > image_height)
    label_anchors[outside_anchors] = -1
    outside_anchors = np.where(
        all_anchors[..., 1] - all_anchors[..., 3] // 2 < 0)
    label_anchors[outside_anchors] = -1
    return label_anchors