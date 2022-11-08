from random import shuffle
from backend import custom_loss_core, SimpleBatchGenerator, define_YOLOv2, set_pretrained_weight, initialize_weight
import numpy as np
import coremltools as ct
import os
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

LABELS = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle', 
          'bus',        'car',      'cat',  'chair',     'cow',
          'diningtable','dog',    'horse',  'motorbike', 'person',
          'pottedplant','sheep',  'sofa',   'train',   'tvmonitor']
LABELS = ['Evidenziatore', 'Gel', 'Matita']

train_img = 'YOLOv2_implementation/training/img/'
train_ann = 'YOLOv2_implementation/training/ann/'

#train_img = '/Users/davideborghini/Desktop/VOCdevkit/VOC2012/JPEGImages/'
#train_ann = '/Users/davideborghini/Desktop/VOCdevkit/VOC2012/Annotations/'

valid_img = 'YOLOv2_implementation/validation/img/'
valid_ann = 'YOLOv2_implementation/validation/ann/'

test_img = 'YOLOv2_implementation/test/img/'
test_ann = 'YOLOv2_implementation/test/ann/'

ANCHORS = np.array([4.968098958333333,2.5675967261904766,
6.160227272727272,4.064346590909091,
2.126862373737374,3.448405934343435,
6.035069444444445,7.6781250000000005,
3.2735624999999997,4.494479166666666,
5.744128787878788,1.216903409090909])


ANCHORS = np.array([5.2297, 1.9902,
                    4.7930,4.0522,
                    2.2934,3.7465,
                    6.0350,7.6781])

BATCH_SIZE = 32
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

from backend import calc_IOU_pred_true_assigned, extract_ground_truth,adjust_scale_prediction, get_cell_grid
def custom_acc(y_true, y_pred):
    true_xy, true_wh, true_box_conf, _ = extract_ground_truth(y_true)
    cell_grid   = get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    pred_xy, pred_wh, _, _ = adjust_scale_prediction(y_pred,cell_grid,ANCHORS)
    return calc_IOU_pred_true_assigned(true_box_conf, true_xy,true_wh,pred_xy,pred_wh)*10.0

def custom_dis(y_true, y_pred):
    true_xy, true_wh, _, _ = extract_ground_truth(y_true)
    cell_grid   = get_cell_grid(GRID_W,GRID_H,BATCH_SIZE,BOX)
    pred_xy, pred_wh, _, _ = adjust_scale_prediction(y_pred,cell_grid,ANCHORS)
    sqDiff = tf.square(true_wh - pred_wh) + tf.square(true_xy - pred_xy)
    return tf.reduce_mean(sqDiff, axis=-1)

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
#optimizer = Adam.Adam()
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
#optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

#model.compile(loss=custom_loss, optimizer=optimizer)
model.compile(loss=custom_loss, optimizer=optimizer, metrics=[custom_acc, custom_dis])

history = model.fit_generator(generator        = train_batch_generator, 
                    steps_per_epoch  = len(train_batch_generator), 
                    epochs           = 5, 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint], 
                    max_queue_size   = 3)

print(history.history)

loss = history.history['loss']
val_loss = history.history['val_loss']
#plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,5.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

acc = history.history['custom_acc']
val_acc = history.history['val_custom_acc']
plt.plot(acc, label='Training Loss')
plt.plot(val_acc, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,0.2])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

dis = history.history['custom_dis']
val_dis = history.history['val_custom_dis']
plt.plot(dis, label='Training Loss')
plt.plot(val_dis, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,82])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']


model.save('model_716.hdf5')