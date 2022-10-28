from random import shuffle
from backend import custom_loss_core, SimpleBatchGenerator, define_YOLOv2, set_pretrained_weight, initialize_weight
import numpy as np
import coremltools as ct
import os
import tensorflow as tf
print(tf.__version__)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']
LABELS = ['Evidenziatore', 'Gel', 'Matita']
LABELS = ['Gel']

train_img = 'YOLOv2_implementation/training/img/'
train_ann = 'YOLOv2_implementation/training/ann/'

#train_img = '/Users/davideborghini/Desktop/VOCdevkit/VOC2012/JPEGImages/'
#train_ann = '/Users/davideborghini/Desktop/VOCdevkit/VOC2012/Annotations/'

valid_img = 'YOLOv2_implementation/validation/img/'
valid_ann = 'YOLOv2_implementation/validation/ann/'

test_img = 'YOLOv2_implementation/test/img/'
test_ann = 'YOLOv2_implementation/test/ann/'

ANCHORS = np.array([1.07709888,  1.78171903,  # anchor box 1, width , height
                    2.71054693,  5.12469308,  # anchor box 2, width,  height
                   10.47181473, 10.09646365,  # anchor box 3, width,  height
                    5.48531347,  8.11011331]) # anchor box 4, width,  height
BATCH_SIZE = 16
BOX = int(len(ANCHORS)/2)
CLASS = len(LABELS)
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT    = 5.0
LAMBDA_COORD     = 1.0
LAMBDA_CLASS     = 1.0
IMAGE_H = 416
IMAGE_W = 416
GRID_H = 13
GRID_W = 13
TRUE_BOX_BUFFER = 50
generator_config = {
    'IMAGE_H'         : IMAGE_H, 
    'IMAGE_W'         : IMAGE_W,
    'GRID_H'          : GRID_H,  
    'GRID_W'          : GRID_W,
    'LABELS'          : LABELS,
    'ANCHORS'         : ANCHORS,
    'BATCH_SIZE'      : BATCH_SIZE,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}

np.random.seed(1)
from backend import parse_annotation
train_image, seen_train_labels = parse_annotation(train_ann,
                                                  train_img, 
                                                  labels=LABELS)
print("N train = {}".format(len(train_image)))
valid_image, seen_valid_labels = parse_annotation(valid_ann,
                                                  valid_img,
                                                  labels=LABELS)
print("N valid = {}".format(len(valid_image)))

def normalize(image):
    return image/255.
train_batch_generator = SimpleBatchGenerator(train_image, generator_config,
                                             norm=normalize, shuffle=True)
valid_batch = SimpleBatchGenerator(valid_image, generator_config, norm=normalize, shuffle=True)
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, 
                                  trainable=False)
model.summary()

path_to_weight = "./yolov2.weights"
nb_conv        = 23
model          = set_pretrained_weight(model,nb_conv, path_to_weight)
layer          = model.layers[-4] # the last convolutional layer
initialize_weight(layer,sd=1/(GRID_H*GRID_W))

def custom_loss(y_true, y_pred):
    loss = custom_loss_core(y_true,
                     y_pred,
                     true_boxes,
                     GRID_W,
                     GRID_H,
                     BATCH_SIZE,
                     ANCHORS,
                     LAMBDA_COORD,
                     LAMBDA_CLASS,
                     LAMBDA_NO_OBJECT, 
                     LAMBDA_OBJECT)
    return loss

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import adam_v2 as Adam

dir_log = "logs/"
try:
    os.makedirs(dir_log)
except:
    pass


generator_config['BATCH_SIZE'] = BATCH_SIZE
early_stop = EarlyStopping(monitor='loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
checkpoint = ModelCheckpoint('weights_yolo_on_voc2012.h5', 
                             monitor='loss', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='min', 
                             period=1)


optimizer = Adam.Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=custom_loss, optimizer=optimizer)

model.fit_generator(generator        = train_batch_generator, 
                    steps_per_epoch  = len(train_batch_generator), 
                    epochs           = 100, 
                    verbose          = 1,
                    #validation_data  = valid_batch,
                    #validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint], 
                    max_queue_size   = 3)

model.save('model_716.hdf5')