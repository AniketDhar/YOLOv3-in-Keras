'''
Darknet-52 for YOLOv3
Created on: 26th June, 2019
Created by: aniket.dhar@topic.nl
https://github.com/Basasuya/basasuya-yolo3/blob/master/yolo3/model.py
'''

import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Input, Activation, add, GlobalAveragePooling2D, Dense, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


def Conv2D_BN_Leaky(x, filters, kernels, strides=1):
    """Conv2D_BN_Leaky
    This function defines a 2D convolution operation followed by BN and LeakyReLU.
    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernels: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and
            height. Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """
    #padding = 'valid' if strides==2 else 'same'
    padding = 'same'
    
    x = Conv2D(filters, kernels,
               padding=padding,
               strides=strides,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def residual_unit(inputs, filters):
    """Residual Unit
    This function defines a series of residual block operations
    # Arguments
        inputs: Tensor, input tensor of residual block.
        filters: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
    # Returns
        Output tensor.
    """
    
    x = Conv2D_BN_Leaky(inputs, filters//2, (1, 1))
    x = Conv2D_BN_Leaky(x, filters, (3, 3))
    x = add([inputs, x])
    #x = Activation('linear')(x)
    
    return x


def residual_block(x, filters, num_blocks):
    """Residual Block
    This function defines a series of residual block operations
    # Arguments
        x: Tensor, input tensor of residual block.
        filters: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        num_blocks: An integer specifying number of residual units
    # Returns
        Output tensor.
    """
    
    x = Conv2D_BN_Leaky(x, filters, (3, 3), strides=2)
    
    for i in range(num_blocks):
        x = residual_unit(x, filters)

    return x


def darknet_body(x):
    """Darknet body having 52 Conv2D layers"""
    x = Conv2D_BN_Leaky(x, 32, (3,3))  #3
    x = residual_block(x, 64, 1)       #10
    x = residual_block(x, 128, 2)      #10*2
    x = residual_block(x, 256, 8)      #10*8
    x = residual_block(x, 512, 8)      #10*8
    x = residual_block(x, 1024, 4)     #10*4
    
    return x


def darknet_classifier():
    """Darknet-52 classifier"""
    inputs = Input(shape=(416, 416, 3))
    x = darknet_body(inputs)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(inputs, x)

    return model


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = Conv2D_BN_Leaky(x, num_filters, (1,1))
    x = Conv2D_BN_Leaky(x, num_filters*2, (3,3))
    x = Conv2D_BN_Leaky(x, num_filters, (1,1))
    x = Conv2D_BN_Leaky(x, num_filters*2, (3,3))
    x = Conv2D_BN_Leaky(x, num_filters, (1,1))
    y = Conv2D_BN_Leaky(x, num_filters*2, (3,3))
    y = Conv2D(out_filters, (1,1), activation='linear', kernel_regularizer=l2(5e-4))(y)
    return x, y


def yolo_body(num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(shape=(416, 416, 3))
    
    darknet = Model(inputs, darknet_body(inputs)) #223
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5)) #19
    
    print(x.shape)  #(None, 13, 13, 512)
    print(y1.shape) #(None, 13, 13, 60)
    
    x = Conv2D_BN_Leaky(x, 256, (1,1))
    x = UpSampling2D(2)(x)
    
    print(x.shape)  #(None, 26, 26, 256)
    
    x = Concatenate()([x, darknet.get_layer('add_19').output])
    print(x.shape)  #(None, 26, 26, 768)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
    
    print(x.shape)  #(None, 26, 26, 256)
    print(y2.shape) #(None, 26, 26, 60)
    
    x = Conv2D_BN_Leaky(x, 128, (1,1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.get_layer('add_10').output])
    print(x.shape)  #(None, 26, 26, 384)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
    
    print(x.shape)  #(None, 26, 26, 128)
    print(y2.shape) #(None, 26, 26, 60)
    
    return Model(inputs, [y1, y2, y3])


def yolo_head(feats, anchors, num_classes, input_shape):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    box_xy = K.sigmoid(feats[..., :2])
    box_wh = K.exp(feats[..., 2:4])
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (box_xy + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = box_wh * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1], # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(3):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_code reletive to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
    '''
    num_classes = 11
    input_shape = (416, 416)
    anchors = np.array(((10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)))
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    input_shape = np.array(input_shape)
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(3)]

    box_data = np.array(box_data, dtype='float32')
    boxes_xy = box_data[..., 0:2]*input_shape[::-1]
    boxes_wh = box_data[..., 2:4]*input_shape[::-1]
    #print(boxes_xy)
    #print(boxes_wh)
    #print(box_data[0, 0])


    m = box_data.shape[0]
    print('box_data.shape[0] is {}'.format(m))

    #print(np.floor(box_data[0, 0]*grid_shapes[0][1]).astype('int32'))
    #print(np.floor(box_data[0, 1]*grid_shapes[0][0]).astype('int32'))

    #for l in range(3):
    #    print(grid_shapes[l][0], grid_shapes[l][1])
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5+num_classes), dtype='float32') for l in range(3)]
    #for i in range(len(y_true)):
    #    print(y_true[i].shape)

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        #print(iou)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)
        print(best_anchor)

        for t, n in enumerate(best_anchor):
            for l in range(3):
                print(l)
                if n in anchor_mask[l]:
                    print('Ahoy!!')
                    i = np.floor(box_data[b, 0]*grid_shapes[l][1]).astype('int32')
                    j = np.floor(box_data[b, 1]*grid_shapes[l][0]).astype('int32')
                    n = anchor_mask[l].index(n)
                    print(b,j,i,n)
                    c = box_data[b, 4].astype('int32')
                    y_true[l][b, j, i, n, 0:4] = box_data[b, 0:4]
                    y_true[l][b, j, i, n, 4] = 1
                    y_true[l][b, j, i, n, 5+c] = 1
                    print(y_true[l][b, j, i, n, :])
                    break
    return y_true


def box_iou(b1, b2):
    '''Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh
    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    '''Return yolo_loss tensor
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(T, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    '''
    yolo_outputs = args[:3]
    y_true = args[3:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(3)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0]

    for l in range(3):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        pred_xy, pred_wh, pred_confidence, pred_class_probs = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet box loss.
        xy_delta = (y_true[l][..., :2]-pred_xy)*grid_shapes[l][::-1]
        wh_delta = K.log(y_true[l][..., 2:4]) - K.log(pred_wh)
        # Avoid log(0)=-inf.
        wh_delta = K.switch(object_mask, wh_delta, K.zeros_like(wh_delta))
        box_delta = K.concatenate([xy_delta, wh_delta], axis=-1)
        box_delta_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        box_loss = object_mask * K.square(box_delta*box_delta_scale)
        confidence_loss = object_mask * K.square(1-pred_confidence) + \
            (1-object_mask) * K.square(0-pred_confidence) * ignore_mask
        class_loss = object_mask * K.square(true_class_probs-pred_class_probs)
        loss += K.sum(box_loss) + K.sum(confidence_loss) + K.sum(class_loss)
    return loss / K.cast(m, K.dtype(loss))



if __name__ == '__main__':
    model = yolo_body(4, 10)
    print(model.summary())
    print(model.output_shape)
