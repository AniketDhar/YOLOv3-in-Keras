'''
Darknet-52 for YOLOv3
Created on: 26th June, 2019
Created by: aniket dhar
'''

import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Input, Activation, add, GlobalAveragePooling2D, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2


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
    x = Conv2D_BN_Leaky(x, 32, (3,3))
    x = residual_block(x, 64, 1)
    x = residual_block(x, 128, 2)
    x = residual_block(x, 256, 8)
    x = residual_block(x, 512, 8)
    x = residual_block(x, 1024, 4)
    
    return x


def darknet():
    """Darknet-52 classifier"""
    inputs = Input(shape=(416, 416, 3))
    x = darknet_body(inputs)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    model = Model(inputs, x)

    return model


if __name__ == '__main__':
    model = darknet()
print(model.summary())
