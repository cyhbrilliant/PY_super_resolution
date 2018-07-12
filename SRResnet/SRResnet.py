import numpy as np
import h5py
import cv2
import tensorflow as tf
import matplotlib.pyplot as mp
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
'''
FSRCNN 915
两次deconv
'''

datasetpath='E:/cuiyuhao/python/dataset/'
file=h5py.File(datasetpath+'Std91_96.h5','r')
Train_data = file['Data'][:]
file.close()

# datasetpath='E:/cuiyuhao/python/dataset/'
# file=h5py.File(datasetpath+'imagenetBGR35W.h5','r')
# Train_data = file['Input'][:]
# file.close()

BNtrain=True
batch_num = 16
Scale=4
PatchSize=96
PatchSize_HF=int(PatchSize/2)
H=int(PatchSize/Scale)
W=int(PatchSize/Scale)

def getBatch(Batch_num):
    Input=[]
    Label=[]
    for i in range(Batch_num):
        index=np.random.randint(0,Train_data.shape[0])
        Label.append(Train_data[index,:,:,:])
        Input.append(cv2.resize(Train_data[index,:,:,:],dsize=(H,W)))
    return np.array(Input),np.array(Label)

# def getBatch_patch(Batch_num):
#     Input=[]
#     Label=[]
#     for i in range(Batch_num):
#         index=np.random.randint(0,Train_data.shape[0])
#         indexX=np.random.randint(0,Train_data.shape[1]-PatchSize)
#         indexY = np.random.randint(0, Train_data.shape[2] - PatchSize)
#         Patch=Train_data[index,indexX:indexX+PatchSize,indexY:indexY+PatchSize,:]
#         Label.append(Patch)
#         Input.append(cv2.resize(Patch,dsize=(H,W)))
#     return np.array(Input),np.array(Label)


#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)

Xp=tf.placeholder(tf.float32,shape=[None,None,None,3])
Yp=tf.placeholder(tf.float32,shape=[None,None,None,3])

Wconv1=weight_variable([5,5,3,64])
Bconv1=bias_variable([64])
conv1=tf.nn.relu(tf.nn.conv2d(Xp,Wconv1,strides=[1,1,1,1],padding='SAME')+Bconv1)

#残差结构
WRes1 = weight_variable([3, 3, 64, 64])
BRes1 = bias_variable([64])
Res1 = tf.nn.relu(batch_norm(tf.nn.conv2d(conv1, WRes1, strides=[1, 1, 1, 1], padding='SAME') + BRes1, BNtrain))
# conv
WRes2 = weight_variable([3, 3, 64, 64])
BRes2 = bias_variable([64])
Res2 = batch_norm(tf.nn.conv2d(Res1, WRes2, strides=[1, 1, 1, 1], padding='SAME') + BRes2, BNtrain)
Res2 = conv1+Res2

for reslayer in range(15):
    #conv+relu
    weightRes1 = weight_variable([3,3,64,64])
    biasRes1 = bias_variable([64])
    convRes = tf.nn.relu(batch_norm(tf.nn.conv2d(Res2,weightRes1,strides=[1,1,1,1],padding='SAME') + biasRes1,BNtrain))
    #conv
    weightRes2 = weight_variable([3,3,64,64])
    biasRes2 = bias_variable([64])
    convRes =batch_norm(tf.nn.conv2d(convRes,weightRes2,strides=[1,1,1,1],padding='SAME') + biasRes2,BNtrain)
    Res2 = convRes + Res2

WRes3 = weight_variable([3, 3, 64, 64])
BRes3 = bias_variable([64])
Res3 = batch_norm(tf.nn.conv2d(Res2, WRes3, strides=[1, 1, 1, 1], padding='SAME') + BRes3, BNtrain)
Res3=conv1+Res3

WDEConv1=weight_variable([9,9,256,64])
BDEConv1=bias_variable([256])
DEConv1=tf.nn.relu(tf.nn.conv2d_transpose(Res3,WDEConv1,output_shape=[batch_num,H*2,W*2,256],strides=[1,2,2,1],padding='SAME')+BDEConv1)

WDEConv2=weight_variable([9,9,3,256])
BDEConv2=bias_variable([3])
OUT=tf.nn.conv2d_transpose(DEConv1,WDEConv2,output_shape=[batch_num,H*Scale,W*Scale,3],strides=[1,2,2,1],padding='SAME')+BDEConv2

loss=tf.reduce_mean(tf.square(Yp-OUT))
TrainStep=tf.train.AdamOptimizer(0.000001).minimize(loss)
#
#
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess,'SRResnet_session/SRResnet_MODE_Std91.ckpt')
for iter in range(100000):
    print('\n',iter)
    Data_batch,Label_batch=getBatch(batch_num)
    error,result=sess.run([loss,TrainStep],feed_dict={Xp:Data_batch,Yp:Label_batch})
    print(error)
    if (iter+1)%1000==0:
        path = saver.save(sess,'SRResnet_session/SRResnet_MODE_Std91_lr.ckpt')
        print(path)