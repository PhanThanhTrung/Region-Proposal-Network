import numpy as np

def normalize_box(bbox):
    left,bottom,width,height=bbox
    center_x,center_y=left+width//2, bottom+height//2
    return np.array(center_x,center_y,width, height)

def IOU(bbox1, bbox2):
    box1_x1,box1_y1,box1_x2,box1_y2=bbox1[0]-bbox1[2]//2,bbox1[0]+bbox1[2]//2, bbox1[1]-bbox1[3]//2,bbox1[1]+bbox1[3]//2
    box2_x1,box2_y1,box2_x2,box2_y2=bbox2[0]-bbox2[2]//2,bbox2[0]+bbox2[2]//2, bbox2[1]-bbox2[3]//2,bbox2[1]+bbox2[3]//2

    overlap_x=min(box1_x2,box2_x2) - max(box1_x1,box2_x1,0)
    overlap_y=min(box1_y2,box2_y2) - max(box1_y1,box2_y1,0)
    overlap=overlap_x*overlap_y

    union=bbox1[2]*bbox1[3]+bbox2[2]*bbox2[3]-overlap

    return overlap/union

def anchor_generator():