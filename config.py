#Data loader configuration
train_image_folder='/home/fluoxetine/Repository/Datasets/Wider_Face/WIDER_train/'
train_anno_file_path='/home/fluoxetine/Repository/Datasets/Wider_Face/wider_face_split/wider_face_train_bbx_gt.txt'
val_image_folder='/home/fluoxetine/Repository/Datasets/Wider_Face/WIDER_val/'
val_anno_file_path='/home/fluoxetine/Repository/Datasets/Wider_Face/wider_face_split/wider_face_val_bbx_gt.txt'

#Region Proposal Network configuration
number_of_anchor=9
backbone='VGG'

#anchors matching configuration
batch_size=256
higher_threshold=0.7
lower_theshold=0.3

#loss configuration
lamda=10 #by default
epsilon=1.0 #for smooth L1