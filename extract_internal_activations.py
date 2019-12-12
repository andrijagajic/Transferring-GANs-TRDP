#from __future__ import absolute_import, division, print_function

import os, sys
sys.path.append(os.getcwd())

import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.save_images
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.cond_batchnorm
import matplotlib.pyplot as plt
import math
import tflib.ops.deconv2d
\import tflib.lsun_label
import tflib.ops.layernorm
import tflib.plot
import pickle

layers_gen = []
layers_dis = []
gener = False

def getActivations(layer,stimuli):
    units = session.run(layer,feed_dict={all_real_data_conv:stimuli})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[1]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,i,:,:], interpolation="nearest", cmap="gray")
if not gener:
    BATCH_SIZE = 600
if gener:
    BATCH_SIZE = 450 # Batch size. Must be a multiple of N_GPUS
    
TARGET_NAME = 'IPCV'
DATA_DIR = 'data/Datasets/' + TARGET_NAME
RESULT_DIR = './result/whole_set6/'
SAMPLES_DIR = RESULT_DIR + '/samples/'
MODEL_DIR = RESULT_DIR + '/model/'

TARGET_DOMAIN = 'IPCV' # Name of target domain 
SOURCE_DOMAIN = 'imagenet' # imagenet, places, celebA, bedroom,
ACGAN = True 
if ACGAN: 
    PRETRAINED_MODEL = './transfer_model/conditional/%s/wgan-gp.model' %SOURCE_DOMAIN 
else: 
    PRETRAINED_MODEL = './transfer_model/unconditional/%s/wgan-gp.model' %SOURCE_DOMAIN 

if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
SAVE_SAMPLES_STEP = 50 # Generate and save samples every SAVE_SAMPLES_STEP
CHECKPOINT_STEP = 4000

ITER_START = 0


ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 1. # How to scale generator's ACGAN loss relative to WGAN loss

N_CLASSES = 10
DIM = 64 # Model dimensionality
N_PIXELS = 64

# Settings for TTUR and orig
CRITIC_ITERS = 5 # How many iterations to train the critic for
D_LR = 0.00001
G_LR = 0.00001
BETA1_D = 0.0
BETA1_G = 0.0
#FID_STEP = 333 # FID evaluation every FID_STEP
FID_STEP = 250 # FID evaluation every FID_STEP
ITERS = 25009 # How many iterations to train for


# Switch on and off batchnormalizaton for the discriminator
# and the generator. Default is on for both.
BN_D=True
BN_G=True

# Log subdirectories are automatically created from
# the above settings and the current timestamp.
N_GPUS = 1 # Number of GPUs
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = N_PIXELS * N_PIXELS * 3 # Number of pixels in each iamge


# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
  print(("*** create sample dir %s" % SAMPLES_DIR))
  os.makedirs(SAMPLES_DIR)
if not os.path.exists(MODEL_DIR):
  print(("*** create checkpoint dir %s" % MODEL_DIR))
  os.makedirs(MODEL_DIR)

#lib.print_model_settings(locals().copy(), LOG_DIR)
lib.print_model_settings(locals().copy())

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """
    # For actually generating decent samples, use this one
    return GoodGenerator, GoodDiscriminator


DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs, labels = None):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:        
        if labels is not None and ACGAN:
            return lib.ops.cond_batchnorm.Batchnorm(name,axes,inputs,labels=labels,n_labels=N_CLASSES)
        else:
            return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)
        
def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True, bn=False, labels = None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if bn:
      output = Normalize(name+'.BN1', [0,2,3], output, labels = labels)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if bn:
      output = Normalize(name+'.BN2', [0,2,3], output, labels = labels)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


# ! Generators



def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu, bn=BN_G, labels = None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    ## supports 32x32 images
    fact = N_PIXELS // 16
    output = lib.ops.linear.Linear('Generator.Input', 128, fact*fact*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, fact, fact])
    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up', bn=bn, labels = labels)
    if gener and len(layers_gen)==0:
        layers_gen.append(output)
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up', bn=bn, labels = labels)
    if gener and len(layers_gen)==1:
        layers_gen.append(output)
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up', bn=bn, labels = labels)
    if gener and len(layers_gen)==2:
        layers_gen.append(output)
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up', bn=bn, labels = labels)
    if bn:
      output = Normalize('Generator.OutputN', [0,2,3], output, labels = labels)
    if gener and len(layers_gen)==3:
        layers_gen.append(output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    if gener and len(layers_gen)==4:
        layers_gen.append(output)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])
# ! Discriminators



def GoodDiscriminator(inputs, dim=DIM, bn=BN_D):
    fact = N_PIXELS // 16

    output = tf.reshape(inputs, [-1, 3, N_PIXELS, N_PIXELS])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)
    if not gener:
        layers_dis.append(output)
    
    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down', bn=bn)
    if not gener:
        layers_dis.append(output)
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down', bn=bn)
    if not gener:
        layers_dis.append(output)
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down', bn=bn)
    if not gener:
        layers_dis.append(output)
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down', bn=bn)
    if not gener:   
        layers_dis.append(output)

    output = tf.reshape(output, [-1, fact*fact*8*dim])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', fact*fact*8*dim, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', fact*fact*8*dim, N_CLASSES, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None
Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, N_PIXELS, N_PIXELS])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    for device_index, (device, real_data_conv, real_labels) in enumerate(zip(DEVICES, split_real_data_conv, labels_splits)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
            fake_labels = tf.cast(tf.random_uniform([BATCH_SIZE//len(DEVICES)])*10, tf.int32)            
            fake_data = Generator(BATCH_SIZE//len(DEVICES), bn=BN_G, labels = fake_labels)

            disc_real, disc_real_acgan = Discriminator(real_data)
            #disc_fake, disc_fake_acgan = Discriminator(fake_data)

            alpha = tf.random_uniform(shape=[BATCH_SIZE//len(DEVICES),1], minval=0., maxval=1. )
            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            
    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    fixed_noise_samples = Generator(100, labels = fixed_labels, noise=fixed_noise)

    def generate_image(frame):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, N_PIXELS, N_PIXELS)), SAMPLES_DIR+'samples_{}.png'.format(frame))


    fake_labels_100 = tf.cast(tf.random_uniform([100])*N_CLASSES, tf.int32)
    samples_100 = Generator(100, labels = fake_labels_100)

    train_gen, dev_gen = lib.lsun_label.load(BATCH_SIZE, DATA_DIR, NUM_TRAIN = 10000)

    def inf_train_gen():
        while True:
            for (images,labels) in train_gen():
                yield images,labels
    gen = inf_train_gen()

    session.run(tf.global_variables_initializer())
    # just save and restore all parameters,instead of learning rate 
    if gener:
        ckpt_saver = tf.train.Saver(lib.params_with_name('Generator'))
    else:
        ckpt_saver = tf.train.Saver(lib.params_with_name('Discriminator.'))
    ckpt_saver.restore(session, PRETRAINED_MODEL)
    ckpt_saver = tf.train.Saver()
    _images, _labels = next(gen)
    if not gener:
        res = session.run(layers_dis, feed_dict={all_real_data_conv: _images, all_real_labels: _labels})
        with open("activations/Bordeaux/" + SOURCE_DOMAIN + "_source/" + TARGET_NAME + "/activations.txt", "wb") as fp:   #Pickling
            pickle.dump(res, fp)
    else:
        res2 = session.run(layers_gen, feed_dict={all_real_data_conv: _images, all_real_labels: _labels})
        with open("activations/Bordeaux/" + SOURCE_DOMAIN +  "_source/" + TARGET_NAME + "/activations_gen.txt", "wb") as fp:   #Pickling
            pickle.dump(res2, fp)