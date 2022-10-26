import AnchorBoxClustering
import InOut
import Loss
import Model
import numpy as np

ANCHORS = np.array([1.07709888,  1.78171903,
                    2.71054693,  5.12469308,
                   10.47181473, 10.09646365,
                    5.48531347,  8.11011331])

LABELS = ['Dispositivo']

train_img = 'YOLOv2_implementation/training/img/'
train_ann = 'YOLOv2_implementation/training/ann/'

valid_img = 'YOLOv2_implementation/validation/img/'
valid_ann = 'YOLOv2_implementation/validation/ann/'

test_img = 'YOLOv2_implementation/test/img/'
test_ann = 'YOLOv2_implementation/test/ann/'