import numpy as np
import h5py
import cv2
import tensorflow as tf
import matplotlib.pyplot as mp
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
'''
FSRCNN 915 test
两次deconv
'''

BNtrain=False
Scale=4
# testimg_Label=mp.imread('1.jpg')
# testimg_Label=mp.imread('butterfly_GT.bmp')
# testimg_Label=mp.imread('lenna.bmp')
# testimg_Label=mp.imread('0.bmp')
testimg_Label=mp.imread('4.bmp')
testimg_Label=cv2.resize(testimg_Label,dsize=(int(testimg_Label.shape[1]/4)*4,int(testimg_Label.shape[0]/4)*4))/127.5-1.
# testimg_Label=cv2.resize(testimg_Label,dsize=(1016,912))
testimg_Data=cv2.resize(testimg_Label,dsize=(int(testimg_Label.shape[1]/Scale),int(testimg_Label.shape[0]/Scale)),interpolation=cv2.INTER_CUBIC)
print(testimg_Data.shape,testimg_Label.shape)
H=int(testimg_Data.shape[0])
W=int(testimg_Data.shape[1])

#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def Leakyrelu(x, alpha=0.2, max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

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
Resize=tf.image.resize_bicubic(Xp,size=[H*Scale,W*Scale])

Wconv1 = weight_variable([5, 5, 3, 64])
Bconv1 = bias_variable([64])
conv1 =Leakyrelu(tf.nn.conv2d(Xp, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + Bconv1)

# 残差结构
WRes1 = weight_variable([3, 3, 64, 64])
BRes1 = bias_variable([64])
Res1 = Leakyrelu(batch_norm(tf.nn.conv2d(conv1, WRes1, strides=[1, 1, 1, 1], padding='SAME') + BRes1, BNtrain))
# conv
WRes2 = weight_variable([3, 3, 64, 64])
BRes2 = bias_variable([64])
Res2 = batch_norm(tf.nn.conv2d(Res1, WRes2, strides=[1, 1, 1, 1], padding='SAME') + BRes2, BNtrain)
Res2 = conv1 + Res2

varlist = [Wconv1,Bconv1,WRes1,BRes1,WRes2,BRes2]
for reslayer in range(9):
    # conv+relu
    weightRes1 = weight_variable([3, 3, 64, 64])
    biasRes1 = bias_variable([64])
    convRes = Leakyrelu(batch_norm(tf.nn.conv2d(Res2, weightRes1, strides=[1, 1, 1, 1], padding='SAME') + biasRes1, BNtrain))
    # conv
    weightRes2 = weight_variable([3, 3, 64, 64])
    biasRes2 = bias_variable([64])
    convRes = batch_norm(tf.nn.conv2d(convRes, weightRes2, strides=[1, 1, 1, 1], padding='SAME') + biasRes2,BNtrain)
    Res2 = convRes + Res2

    varlist.append(weightRes1)
    varlist.append(biasRes1)
    varlist.append(weightRes2)
    varlist.append(biasRes2)

WRes3 = weight_variable([3, 3, 64, 64])
BRes3 = bias_variable([64])
Res3 = batch_norm(tf.nn.conv2d(Res2, WRes3, strides=[1, 1, 1, 1], padding='SAME') + BRes3, BNtrain)
Res3 = conv1 + Res3
varlist.append(WRes3)
varlist.append(BRes3)

DEConv1 =tf.image.resize_nearest_neighbor(Res3,size=[H*2,W*2])

# *4
Wconv2 = weight_variable([3, 3, 64, 64])
Bconv2 = bias_variable([64])
conv2 = Leakyrelu(tf.nn.conv2d(DEConv1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + Bconv2)
varlist.append(Wconv2)
varlist.append(Bconv2)

for reslayer in range(6):
    # conv+relu
    weightRes1 = weight_variable([3, 3, 64, 64])
    biasRes1 = bias_variable([64])
    convRes = Leakyrelu(batch_norm(tf.nn.conv2d(conv2, weightRes1, strides=[1, 1, 1, 1], padding='SAME') + biasRes1, BNtrain))
    # conv
    weightRes2 = weight_variable([3, 3, 64, 64])
    biasRes2 = bias_variable([64])
    convRes = batch_norm(tf.nn.conv2d(convRes, weightRes2, strides=[1, 1, 1, 1], padding='SAME') + biasRes2, BNtrain)
    conv2 = convRes + conv2

    varlist.append(weightRes1)
    varlist.append(biasRes1)
    varlist.append(weightRes2)
    varlist.append(biasRes2)

WRes4 = weight_variable([3, 3, 64, 256])
BRes4 = bias_variable([256])
Res4 = Leakyrelu(batch_norm(tf.nn.conv2d(conv2, WRes4, strides=[1, 1, 1, 1], padding='SAME') + BRes4, BNtrain))
varlist.append(WRes4)
varlist.append(BRes4)

DEConv2=tf.image.resize_nearest_neighbor(Res4,size=[H * Scale, W * Scale])

Wconv3=weight_variable([3,3,256,64])
Bconv3=bias_variable([64])
conv3 = Leakyrelu(tf.nn.conv2d(DEConv2, Wconv3, strides=[1, 1, 1, 1], padding='SAME') + Bconv3)
varlist.append(Wconv3)
varlist.append(Bconv3)

Wconv4 = weight_variable([3, 3, 64, 3])
Bconv4 = bias_variable([3])
OUT = (tf.nn.tanh(tf.nn.conv2d(conv3, Wconv4, strides=[1, 1, 1, 1], padding='SAME') + Bconv4) + 1) * 127.5


# OUT = tf.nn.conv2d_transpose(DEConv1, WDEConv2, output_shape=[1, H * Scale, W * Scale, 3],strides=[1, 2, 2, 1], padding='SAME') + BDEConv2


loss=tf.reduce_mean(tf.square(Yp-(OUT/127.5-1)))
# TrainStep=tf.train.AdamOptimizer(0.0001).minimize(loss)
bicubicloss=tf.reduce_mean(tf.square(Yp-Resize))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet_Full.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/WSRGAN_VGG22_imagenet_Full.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full2.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full3.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full4.ckpt')#lossG
# saver.restore(sess,'UpscaleSRGAN_session/WSRGAN_VGG22_imagenet_Full4.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full5.ckpt')#lossLSR
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full5x.ckpt')#lossLSR
saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full5xx.ckpt')#lossLSR


iter=45
errbicubic,bicubic,testimg,err=sess.run([bicubicloss,Resize,OUT,loss], feed_dict={Xp: testimg_Data[np.newaxis,:,:,:], Yp: testimg_Label[np.newaxis,:,:,:]})
testimg=cv2.cvtColor(testimg[0,:,:,:],cv2.COLOR_RGB2BGR)
bicubic=cv2.cvtColor(bicubic[0,:,:,:],cv2.COLOR_RGB2BGR)
cv2.imwrite('OUTPUT/'+str(iter)+'.jpg',testimg)
cv2.imwrite('OUTPUT/bicubic.jpg',bicubic)
# cv2.imwrite('OUTPUT/1orin.jpg',testimg_Data)
psnr=10*np.log10(4/err)
bicubicpsnr=10*np.log10(4/errbicubic)
print('bicubicpsnr=',bicubicpsnr)
print('psnr=',psnr)
