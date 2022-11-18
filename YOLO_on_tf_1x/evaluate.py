from statistics import mode
import coremltools as ct
import os
from backend import define_YOLOv2
import numpy as np
from backend import ImageReader
import matplotlib.pyplot as plt

LABELS = ['Evidenziatore', 'Gel', 'Matita']

ANCHORS = np.array([5.2297, 1.9902,
                    4.7930,4.0522,
                    2.2934,3.7465,
                    6.0350,7.6781])

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
    'BATCH_SIZE'      : TRUE_BOX_BUFFER,
    'TRUE_BOX_BUFFER' : TRUE_BOX_BUFFER,
}
model, _ = define_YOLOv2(IMAGE_H, IMAGE_W,GRID_H,GRID_W,TRUE_BOX_BUFFER,BOX,CLASS,trainable=False)
model.load_weights("weights_yolo_on_voc2012.h5")
# handle the hack input
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))


class OutputRescaler(object):
    def __init__(self,ANCHORS):
        self.ANCHORS = ANCHORS

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x/np.min(x)*t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)
    def get_shifting_matrix(self,netout):
        
        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[...,0]
        
        ANCHORSw = self.ANCHORS[::2]
        ANCHORSh = self.ANCHORS[1::2]
       
        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:,igrid_w,:] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h,:,:] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):    
            mat_ANCHOR_W[:,:,ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no) 
        for ianchor in range(BOX):    
            mat_ANCHOR_H[:,:,ianchor] = ANCHORSh[ianchor]
        return(mat_GRID_W,mat_GRID_H,mat_ANCHOR_W,mat_ANCHOR_H)

    def fit(self, netout):    
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)
        
        a single image output of model.predict()
        '''
        GRID_H, GRID_W, BOX = netout.shape[:3]
        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = self.get_shifting_matrix(netout)
        # bounding box parameters
        netout[..., 0]   = (self._sigmoid(netout[..., 0]) + mat_GRID_W)/GRID_W # x      unit: range between 0 and 1
        netout[..., 1]   = (self._sigmoid(netout[..., 1]) + mat_GRID_H)/GRID_H # y      unit: range between 0 and 1
        netout[..., 2]   = (np.exp(netout[..., 2]) * mat_ANCHOR_W)/GRID_W      # width  unit: range between 0 and 1
        netout[..., 3]   = (np.exp(netout[..., 3]) * mat_ANCHOR_H)/GRID_H      # height unit: range between 0 and 1
        # rescale the confidence to range 0 and 1 
        netout[..., 4]   = self._sigmoid(netout[..., 4])
        expand_conf      = np.expand_dims(netout[...,4],-1) # (N grid h , N grid w, N anchor , 1)
        # rescale the class probability to range between 0 and 1
        # Pr(object class = k) = Pr(object exists) * Pr(object class = k |object exists)
        #                      = Conf * P^c
        netout[..., 5:]  = expand_conf * self._softmax(netout[..., 5:])
        # ignore the class probability if it is less than obj_threshold 
        return(netout)

from backend import BoundBox
def find_high_class_probability_bbox(netout_scale, obj_threshold):
    '''
    == Input == 
    netout : y_pred[i] np.array of shape (GRID_H, GRID_W, BOX, 4 + 1 + N class)
    
             x, w must be a unit of image width
             y, h must be a unit of image height
             c must be in between 0 and 1
             p^c must be in between 0 and 1
    == Output ==
    
    boxes  : list containing bounding box with Pr(object is in class C) > 0 for at least in one class C 
    
             
    '''
    GRID_H, GRID_W, BOX = netout_scale.shape[:3]
    
    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scale[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scale[row,col,b,:4]
                    confidence = netout_scale[row,col,b,4]
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return(boxes)



from backend import parse_annotation
valid_img = 'YOLOv2_implementation/validation/img/'
valid_ann = 'YOLOv2_implementation/validation/ann/'
valid_image, seen_valid_labels = parse_annotation(valid_ann,
                                                  valid_img, 
                                                  labels=LABELS)

def IoU(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou




def getTrueRect(imgTrueInfo, objName):
    for elem in imgTrueInfo['object']:
        if elem['name'] == objName:
            #modificare qui perché le predizioni sono in spazi con la y ribaltata
            bb = [elem['xmin'], imgTrueInfo['height'] - elem['ymax'], elem['xmax'], imgTrueInfo['height'] - elem['ymin']]
            bb[0] = (bb[0] / imgTrueInfo['width'])*416
            bb[1] = (bb[1] / imgTrueInfo['height'])*416
            bb[2] = (bb[2] / imgTrueInfo['width'])*416
            bb[3] = (bb[3] / imgTrueInfo['height'])*416
            return bb
    return None

def validate():
    model.load_weights("weights_yolo_on_voc2012.h5")
    imageReader = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)
    positiveIoU = [0, 0, 0]
    negativeIoU = [0, 0, 0]
    for imgObj in valid_image:

        #get image info
        location = imgObj['filename']
        out = imageReader.fit(location)
        X_test = np.expand_dims(out,0)

        # get predictions
        y_pred = model.predict([X_test,dummy_array])
        netout         = y_pred[0]
        outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
        netout_scale   = outputRescaler.fit(netout)
        obj_threshold = 0.3
        boxes = find_high_class_probability_bbox(netout_scale,obj_threshold)
        bestBoxes = [None, None, None]
        for box in boxes:
            if bestBoxes[box.label] != None:
                if bestBoxes[box.label].get_score() < box.get_score():
                    bestBoxes[box.label] = box
            else:
                bestBoxes[box.label] = box
        for box in boxes:
            
            trueBox =  getTrueRect(imgObj, LABELS[box.label])
            
            predBox = [box.xmin*IMAGE_W, box.ymin*IMAGE_H, box.xmax*IMAGE_W, box.ymax*IMAGE_H]

            #print(IoU(boxA=predBox, boxB=trueBox))
            if(trueBox != None and IoU(boxA=predBox, boxB=trueBox) > 0.5):
                #print("positive: ", LABELS[box.label])
                positiveIoU[box.label] += 1
            else:
                #print("negative: ", LABELS[box.label])
                negativeIoU[box.label] += 1
    pos = 0
    neg = 0
    for i in range(len(LABELS)):
        pos += positiveIoU[i]
        neg += negativeIoU[i]
        print(LABELS[i].split("z")[0],"\t","{:2.2f}".format(positiveIoU[i]/max((positiveIoU[i] + negativeIoU[i]), 1)), "\t+ ", positiveIoU[i], "\t- ", negativeIoU[i])
    if neg + pos == 0:
        return 0
    return pos/(pos+neg)

print("*"*30)
print(validate())
print("*"*30)

outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
from backend import calc_IOU_pred_true_assigned, extract_ground_truth,adjust_scale_prediction, get_cell_grid
def customIoU(y_pred, y_true):
    true_xy, true_wh, _, class_index = extract_ground_truth(y_true)
    netout         = y_pred[0]
    netout_scale   = outputRescaler.fit(netout)
    obj_threshold = 0.2
    boxes = find_high_class_probability_bbox(netout_scale,obj_threshold)
    