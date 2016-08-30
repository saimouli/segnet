from __future__ import absolute_import
from __future__ import print_function
import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
import keras.layers.containers as containers
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
import keras.models as models
import keras.layers.containers as containers
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, AutoEncoder, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.regularizers import ActivityRegularizer

from keras import backend as K

import cv2
import numpy as np

path = './CamVid/'
data_shape = 360*480
#%%
def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([360,480,12])    
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x
    
def test_data():
    test_data =[]
    import os
    with open(path+'test.txt') as f:
        txt1 = f.readlines()
        txt1 = [line.split(' ') for line in txt1]
    for i in range(len(txt)):
        ## these paths are very specific to my machine
        test_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2)) #gets the images from train folder
    #test_label.append(binarylab(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0])) #images from trainannotate
        print(';',end='')
    return np.array(test_data)
    

test_data = test_data()
#test_label= np.reshape(test_label,(367,data_shape,12))
#%%
#test= normalized(cv2.imread(os.getcwd()+ txt[1][0][7:])) #works with this function only 

def prep_data():
    train_data = []
    train_label = []
    import os
    with open(path+'train.txt') as f:
        txt2 = f.readlines()
        txt2 = [line.split(' ') for line in txt2]
    for i in range(len(txt2)):
        ## these paths are very specific to my machine
        train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt2[i][0][7:])),2)) #gets the images from train folder
        train_label.append(binarylab(cv2.imread(os.getcwd() + txt2[i][1][7:][:-1])[:,:,0])) #images from trainannotate
        print('.',end='')
    return np.array(train_data), np.array(train_label)

train_data, train_label = prep_data()
train_label = np.reshape(train_label,(367,data_shape,12))

# class weigths
class_weighting= [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

class UnPooling2D(Layer):
    """A 2D Repeat layer"""
    def __init__(self, poolsize=(2, 2)):
        super(UnPooling2D, self).__init__()
        self.input = T.tensor4()
        self.poolsize = poolsize

    @property
    def output_shape(self):
        input_shape = self.input_shape
        return (input_shape[0], input_shape[1],
                self.poolsize[0] * input_shape[2],
                self.poolsize[1] * input_shape[3])

    def get_output(self, train):
        X = self.get_input(train)
        s1 = self.poolsize[0]
        s2 = self.poolsize[1]
        output = X.repeat(s1, axis=2).repeat(s2, axis=3)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
            "poolsize":self.poolsize}
    
def create_encoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return [
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),

        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
        Activation('relu'),
    ]

def create_decoding_layers():
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2
    return[
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(512, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UnPooling2D(poolsize=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(256, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UnPooling2D(poolsize=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(128, kernel, kernel, border_mode='valid'),
        BatchNormalization(),

        UnPooling2D(poolsize=(pool_size,pool_size)),
        ZeroPadding2D(padding=(pad,pad)),
        Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
        BatchNormalization(),
    ]
#%%

autoencoder = Sequential()
    # Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
autoencoder.add(Layer(input_shape=(3, 360, 480)))
    
    #autoencoder.add(GaussianNoise(sigma=0.3))
autoencoder.encoding_layers = create_encoding_layers()
autoencoder.decoding_layers = create_decoding_layers()
for l in autoencoder.encoding_layers:
    autoencoder.add(l)
for l in autoencoder.decoding_layers:
    autoencoder.add(l)
autoencoder.add(Convolution2D(12, 1, 1, border_mode='valid',))
autoencoder.add(Reshape((12,data_shape)))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))    
    
#from keras.optimizers import SGD
#optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)

autoencoder.load_weights('model_weight_ep100.hdf5')
autoencoder.compile(loss="categorical_crossentropy", optimizer='adadelta')
#%%

import matplotlib.pyplot as plt
#matplotlib inline

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label[l,0]
        g[temp==l]=label[l,1]
        b[temp==l]=label[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    if plot:
        plt.imshow(rgb)
    else:
        return rgb
#%%
import os
gt = []
with open(path+'test.txt') as f:
    txt = f.readlines()
    txt = [line.split(' ') for line in txt]
    #print (txt)
for i in range(len(txt)):
    # print (txt[i][0][7:])
    #print (txt[i][1][7:][:-1])
    gt.append(cv2.imread(os.getcwd() + txt[i][0][7:]))
#gt.append(cv2.imread(os.getcwd() + txt[60][0][7:]))
#CamVid/test/Seq05VD_f02190.png

output = autoencoder.predict_proba(test_data[2:3])
pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)), False)
plt.imshow(pred)
plt.figure(2)
plt.imshow(gt[2])
