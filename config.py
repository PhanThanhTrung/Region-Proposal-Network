#Data loader configuration
train_image_folder='/Users/hit.fluoxetine/Datasets/Sample_dataset/images/'
train_anno_file_path='/Users/hit.fluoxetine/Datasets/Sample_dataset/annotations/xmls/'
#Region Proposal Network configuration
number_of_anchor=9
backbone='VGG'
output_stride=16
#anchors matching configuration
batch_size=256
higher_threshold=0.7
lower_threshold=0.3
scales=[0.5,1,2]
ratios=[1,2,1,1,2,1]

#loss configuration
lamda=10 #by default
epsilon=1.0 #for smooth L1