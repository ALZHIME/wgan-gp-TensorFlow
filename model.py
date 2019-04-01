import tensorflow as tf
import numpy as np

REGULARIZER_COF = 0.01

def _fc_variable(weight_shape,name):
        with tf.variable_scope(name):
            # check weight_shape
            input_channels  = int(weight_shape[0])
            output_channels = int(weight_shape[1])
            weight_shape    = (input_channels, output_channels)
            regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

            # define variables
            weight = tf.get_variable("w", weight_shape     ,
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer =regularizer)
            bias   = tf.get_variable("b", [weight_shape[1]],
                                    initializer=tf.constant_initializer(0.0))
        return weight, bias

def _conv_variable(weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _deconv_variable(weight_shape,name="deconv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape    ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [input_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _deconv2d(x, W, output_shape, stride=1):
    # x           : [nBatch, height, width, in_channels]
    # output_shape: [nBatch, height, width, out_channels]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

def calcImageSize(dh,dw,stride):
    return int(math.ceil(float(dh)/float(stride))),int(math.ceil(float(dw)/float(stride)))

def loadModel(model_path=None):
    if model_path: saver.restore(sess, model_path)

def _deconv_layer(x,input_layer, output_layer, stride=2, filter_size=5, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*stride,w*stride,output_layer], stride=stride) + deconv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormd"+name)
    h = tf.nn.leaky_relu(h)
    return h

def _conv_layer(x, input_layer, output_layer, stride, filter_size=5, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    #h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormc"+name)
    h = tf.nn.leaky_relu(h)
    return h

def buildGenerator(z, z_dim, img_size, nBatch, reuse=False, isTraining=True):
    with tf.variable_scope("Generator") as scope:
        if reuse: scope.reuse_variables()
        h = z
        # fc1
        g_fc1_w, g_fc1_b = _fc_variable([z_dim,512*8*8],name="fc1")
        h = tf.matmul(h, g_fc1_w) + g_fc1_b
        h = tf.nn.relu(h)
        #
        h = tf.reshape(h,(-1,8,8,512))
        # deconv4
        h = _deconv_layer(h,512,256,name="g4")
        h = _deconv_layer(h,256,128,name="g3")
        h = _deconv_layer(h,128,64,name="g2")
        h = _deconv_layer(h,64,64,name="g1")

        g_deconv1_w, g_deconv1_b = _conv_variable([1,1,64,3],name="deconv1")
        h = _conv2d(h,g_deconv1_w, stride=1) + g_deconv1_b
        y = tf.tanh(h)

    return y

def buildDiscriminator(y, nBatch, reuse=False, isTraining=True):
    with tf.variable_scope("Discriminator") as scope:
        if reuse: scope.reuse_variables()
        h = y
        # conv1
        h = _conv_layer(h,3,64,2,name="d1")
        # conv2
        h = _conv_layer(h,64,128,2,name="d2")
        h = _conv_layer(h,128,256,2,name="d3")
        h = _conv_layer(h,256,512,2,name="d4")
        # fc1
        n_b, n_h, n_w, n_f = [int(x) for x in h.get_shape()]
        h = tf.reshape(h,[nBatch,n_h*n_w*n_f])
        print(h)
        d_fc1_w, d_fc1_b = _fc_variable([n_h*n_w*n_f,1],name="fc1")
        h = tf.matmul(h, d_fc1_w) + d_fc1_b

        ### summary
    return h
