from re import T
import numpy as np
import tensorflow as tf
import time
import os
from AnchorBoxClustering import parse_annotation
from InOut import SimpleBatchGenerator, imgNormalize
from Model import LABELS, define_YOLOv2, set_pretrained_weight, initialize_weight
from Loss import ANCHORS, IMAGE_H, IMAGE_W, GRID_H, GRID_W, TRUE_BOX_BUFFER, BOX, custom_loss_core


BATCH_SIZE = 16
CLASS = len(LABELS)
LAMBDA_NO_OBJECT = 1.0
LAMBDA_OBJECT    = 5.0
LAMBDA_COORD     = 1.0
LAMBDA_CLASS     = 1.0
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

train_img = 'YOLOv2_implementation/training/img/'
train_ann = 'YOLOv2_implementation/training/ann/'

valid_img = 'YOLOv2_implementation/validation/img/'
valid_ann = 'YOLOv2_implementation/validation/ann/'

test_img = 'YOLOv2_implementation/test/img/'
test_ann = 'YOLOv2_implementation/test/ann/'

#np.random.seed(int(time.time()))
train_image, seen_train_lables = parse_annotation(train_ann,
                                                  train_img,
                                                  labels=LABELS)
print("N train immages = {}".format(len(train_image)))
train_batch_generator = SimpleBatchGenerator(train_image, generator_config, norm=imgNormalize, shuffle=True)
model, true_boxes = define_YOLOv2(IMAGE_H,IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS, 
                                  trainable=False)

path_to_weight = "./YOLOv2_implementation/yolov2.weights"
nb_conv        = 22
model          = set_pretrained_weight(model,nb_conv, path_to_weight)
layer          = model.layers[-4] # the last convolutional layer
initialize_weight(layer,sd=1/(GRID_H*GRID_W))


def custom_loss(y_true, y_pred):
    return custom_loss_core(y_true,
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

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop

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
                             save_freq=1)


#optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)


model.compile(loss=custom_loss, optimizer=optimizer)
#model.compile(loss='mse', optimizer=optimizer)
#model.fit(train_batch_generator, steps_per_epoch  = len(train_batch_generator), epochs= 5)
model.fit(train_batch_generator, 
        steps_per_epoch  = len(train_batch_generator), 
        epochs           = 4, 
        verbose          = 1,
        #validation_data  = valid_batch,
        #validation_steps = len(valid_batch),
        callbacks        = [early_stop, checkpoint], 
        max_queue_size   = 3)
model.save('./model_716')