
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, Conv3D, Lambda, Input, Concatenate
from keras.layers import Reshape, AveragePooling3D, LeakyReLU, Activation
import keras.backend as K

n_upsample = 0
n_conv2d_p = 0
n_conv2d_r = 0
n_conv3d_r = 0
n_conv3d_p = 0

def NNUpsampling(scale=2):
    global n_upsample
    n_upsample += 1
    def out_shape(input_shape, scale=scale):
        return (input_shape[0], input_shape[1]*scale, input_shape[2]*scale, input_shape[3])
    def NNresize(x, scale=scale):
        return K.resize_images(x, scale, scale, 'channels_last')
    return Lambda(NNresize, name='nn_upsampling_'+str(n_upsample), output_shape=out_shape)

def conv2D_with_bc(input_shape, filters, kernel_size=(3,3), boundary_condition='reflective'):
    # Output size only consistent with odd kernel size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    kx = int(np.floor(kernel_size[0]/2.0))
    ky = int(np.floor(kernel_size[1]/2.0))

    def periodic_padding_2D(X):
        m = K.concatenate([X[:,-kx:,:], X, X[:,:kx,:]], axis=1)
        t = m[:,:,-ky:]
        b = m[:,:,:ky]
        return K.concatenate([t,m,b], axis=2)

    def reflective_padding_2D(X):
        m = K.concatenate([K.reverse(X[:,:kx], axes=1), X, K.reverse(X[:,-kx:,:], axes=1)], axis=1)
        t = K.reverse(m[:,:,:ky], axes=2)
        b = K.reverse(m[:,:,-ky:], axes=2)
        return K.concatenate([t,m,b], axis=2)

    inp = Input(shape=input_shape)
    if boundary_condition is 'periodic':
        global n_conv2d_p
        n_conv2d_p += 1
        name = 'conv2d_periodic_%d' % n_conv2d_p
        padded = Lambda(periodic_padding_2D)(inp)
    elif boundary_condition is 'reflective':
        global n_conv2d_r
        n_conv2d_r += 1
        name = 'conv2d_reflective_%d' % n_conv2d_r
        padded = Lambda(reflective_padding_2D)(inp)
    else:
        print('Invalid boundary condition')
        return None
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding='valid')(padded)
    return Model(inputs=inp, outputs=conv, name=name)

def conv3D_with_bc(input_shape, filters, kernel_size=(3,3,3), boundary_condition='reflective'):
    # Output size only consistent with odd kernel size

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    kx = int(np.floor(kernel_size[0]/2.0))
    ky = int(np.floor(kernel_size[1]/2.0))
    kz = int(np.floor(kernel_size[2]/2.0))

    def periodic_padding_3D(X):
        m = K.concatenate([X[:,-kx:], X, X[:,:kx]], axis=1)
        m = K.concatenate([m[:,:,-ky:], m, m[:,:,:ky]], axis=2)
        m = K.concatenate([m[:,:,:,-kz:], m, m[:,:,:,:kz]], axis=3)
        return m

    def reflective_padding_3D(X):
        m = K.concatenate([K.reverse(X[:,:kx], axes=1), X, K.reverse(X[:,-kx:,:], axes=1)], axis=1)
        m = K.concatenate([K.reverse(m[:,:,:ky], axes=2), m, K.reverse(m[:,:,-ky:], axes=2)], axis=2)
        m = K.concatenate([K.reverse(m[:,:,:,:kz], axes=3), m, K.reverse(m[:,:,:,-kz:], axes=3)], axis=3)
        return m

    inp = Input(shape=input_shape)
    if boundary_condition is 'periodic':
        global n_conv3d_p
        n_conv3d_p += 1
        name = 'conv3d_periodic_%d' % n_conv3d_p
        padded = Lambda(periodic_padding_3D)(inp)
    elif boundary_condition is 'reflective':
        global n_conv3d_r
        n_conv3d_r += 1
        name = 'conv3d_reflective_%d' % n_conv3d_r
        padded = Lambda(reflective_padding_3D)(inp)
    else:
        print('Invalid boundary condition')
        return None
    conv = Conv3D(filters=filters, kernel_size=kernel_size, padding='valid')(padded)
    return Model(inputs=inp, outputs=conv, name=name)

def create_model(
    n_out=3,
    input_shape=(128,128,10),
    lrelu_factor=0.1,
    boundary_condition='reflective',
    out_labels = None,
    last_relu=True
):
    
    def activation():
        return LeakyReLU(alpha=lrelu_factor)

    if not isinstance(last_relu, list):
        last_relu = [last_relu] * n_out
        
    if out_labels is None:
        out_labels = ['out%d' % i for i in range(n_out)]
    else:
        assert len(out_labels) == n_out
    
    # ==== Common branch
    
    inp = Input(shape=input_shape)
    x = Reshape(input_shape+(1,))(inp)
    
    x = conv3D_with_bc(input_shape=x.shape[1:], filters=4, kernel_size=(3,3,3), boundary_condition=boundary_condition)(x)
    x = activation()(x)
    x = AveragePooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = conv3D_with_bc(input_shape=x.shape[1:], filters=8, kernel_size=(3,3,3), boundary_condition=boundary_condition)(x)
    x = activation()(x)
    x = AveragePooling3D(pool_size=(2,2,2), strides=(2,2,2))(x)

    x = conv3D_with_bc(input_shape=x.shape[1:], filters=16, kernel_size=(3,3,3), boundary_condition=boundary_condition)(x)
    x = activation()(x)
    x = AveragePooling3D(pool_size=(2,2,1), strides=(2,2,1))(x)

    shape = x.shape.as_list()
    x = Reshape(tuple(shape[1:3])+(shape[3]*shape[4],))(x)

    x = conv2D_with_bc(x.shape[1:], filters=64, kernel_size=3, boundary_condition=boundary_condition)(x)
    x = activation()(x)
    x = conv2D_with_bc(x.shape[1:], filters=64, kernel_size=3, boundary_condition=boundary_condition)(x)
    x = activation()(x)

    # ==== Split branches

    outputs = []
    for i in range(n_out):

        h = NNUpsampling(scale=2)(x)
        h = conv2D_with_bc(h.shape[1:], filters=16, kernel_size=3, boundary_condition=boundary_condition)(h)
        h = activation()(h)
        h = conv2D_with_bc(h.shape[1:], filters=16, kernel_size=3, boundary_condition=boundary_condition)(h)
        h = activation()(h)

        h = NNUpsampling(scale=2)(h)
        h = conv2D_with_bc(h.shape[1:], filters=16, kernel_size=3, boundary_condition=boundary_condition)(h)
        h = activation()(h)
        h = conv2D_with_bc(h.shape[1:], filters=16, kernel_size=3, boundary_condition=boundary_condition)(h)
        h = activation()(h)

        h = NNUpsampling(scale=2)(h)
        h = conv2D_with_bc(h.shape[1:], filters=16, kernel_size=3, boundary_condition=boundary_condition)(h)
        h = activation()(h)
        h = conv2D_with_bc(h.shape[1:], filters=16, kernel_size=3, boundary_condition=boundary_condition)(h)
        h = activation()(h)

        h = conv2D_with_bc(h.shape[1:], filters=1, kernel_size=3, boundary_condition=boundary_condition)(h)
        if last_relu[i]:
            h = Activation('relu')(h)

        out = Reshape(input_shape[:2], name=out_labels[i])(h)
        outputs.append(out)

    model = Model(inputs=inp, outputs=outputs)

    return model

