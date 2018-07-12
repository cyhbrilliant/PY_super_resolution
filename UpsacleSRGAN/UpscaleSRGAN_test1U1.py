import numpy as np
import h5py
import cv2
import tensorflow as tf
import matplotlib.pyplot as mp
import vgg19_trainable as vgg19
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
'''
FSRCNN 915 test
两次deconv
'''
iter=6
BNtrain=False
Scale=4

# testimg_Label=mp.imread('1m.jpg')
# testimg_Label=mp.imread('butterfly_GT.bmp')
# testimg_Label=mp.imread('lenna.bmp')
# testimg_Label=mp.imread('10.bmp')
# testimg_Label=mp.imread('3.bmp')
# testimg_Label=cv2.imread('1.png')
testimg_Label=cv2.imread('../dataset/Set14/M'+str(iter)+'.bmp')
# testimg_Label=cv2.imread('../dataset/Medical/M'+str(iter)+'.png')
testimg_Label=cv2.cvtColor(testimg_Label,cv2.COLOR_BGR2RGB)
testimg_Label=cv2.resize(testimg_Label,dsize=(int(testimg_Label.shape[1]/4)*4,int(testimg_Label.shape[0]/4)*4))/127.5-1.
# testimg_Label=cv2.resize(testimg_Label,dsize=(1016,912))
testimg_Data=cv2.resize(testimg_Label,dsize=(int(testimg_Label.shape[1]/Scale),int(testimg_Label.shape[0]/Scale)),interpolation=cv2.INTER_AREA)
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

def Discriminator(GXp,Yp):
    Wconv1=weight_variable([3,3,3,64])
    Bconv1=bias_variable([64])
    GConv1=Leakyrelu(tf.nn.conv2d(GXp,Wconv1,strides=[1,1,1,1],padding='SAME')+Bconv1)
    DConv1=Leakyrelu(tf.nn.conv2d(Yp,Wconv1,strides=[1,1,1,1],padding='SAME')+Bconv1)

    Wconv2=weight_variable([3,3,64,64])
    Bconv2=bias_variable([64])
    GConv2=Leakyrelu(batch_norm(tf.nn.conv2d(GConv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2,BNtrain))
    DConv2=Leakyrelu(batch_norm(tf.nn.conv2d(DConv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2,BNtrain))

    Wconv3=weight_variable([3,3,64,128])
    Bconv3=bias_variable([128])
    GConv3=Leakyrelu(batch_norm(tf.nn.conv2d(GConv2,Wconv3,strides=[1,1,1,1],padding='SAME')+Bconv3,BNtrain))
    DConv3=Leakyrelu(batch_norm(tf.nn.conv2d(DConv2,Wconv3,strides=[1,1,1,1],padding='SAME')+Bconv3,BNtrain))

    Wconv4=weight_variable([3,3,128,128])
    Bconv4=bias_variable([128])
    GConv4=Leakyrelu(batch_norm(tf.nn.conv2d(GConv3,Wconv4,strides=[1,2,2,1],padding='SAME')+Bconv4,BNtrain))
    DConv4=Leakyrelu(batch_norm(tf.nn.conv2d(DConv3,Wconv4,strides=[1,2,2,1],padding='SAME')+Bconv4,BNtrain))

    Wconv5=weight_variable([3,3,128,256])
    Bconv5=bias_variable([256])
    GConv5=Leakyrelu(batch_norm(tf.nn.conv2d(GConv4,Wconv5,strides=[1,1,1,1],padding='SAME')+Bconv5,BNtrain))
    DConv5=Leakyrelu(batch_norm(tf.nn.conv2d(DConv4,Wconv5,strides=[1,1,1,1],padding='SAME')+Bconv5,BNtrain))

    Wconv6=weight_variable([3,3,256,256])
    Bconv6=bias_variable([256])
    GConv6=Leakyrelu(batch_norm(tf.nn.conv2d(GConv5,Wconv6,strides=[1,2,2,1],padding='SAME')+Bconv6,BNtrain))
    DConv6=Leakyrelu(batch_norm(tf.nn.conv2d(DConv5,Wconv6,strides=[1,2,2,1],padding='SAME')+Bconv6,BNtrain))

    Wconv7=weight_variable([3,3,256,512])
    Bconv7=bias_variable([512])
    GConv7=Leakyrelu(batch_norm(tf.nn.conv2d(GConv6,Wconv7,strides=[1,1,1,1],padding='SAME')+Bconv7,BNtrain))
    DConv7=Leakyrelu(batch_norm(tf.nn.conv2d(DConv6,Wconv7,strides=[1,1,1,1],padding='SAME')+Bconv7,BNtrain))

    Wconv8=weight_variable([3,3,512,512])
    Bconv8=bias_variable([512])
    GConv8=Leakyrelu(batch_norm(tf.nn.conv2d(GConv7,Wconv8,strides=[1,2,2,1],padding='SAME')+Bconv8,BNtrain))
    DConv8=Leakyrelu(batch_norm(tf.nn.conv2d(DConv7,Wconv8,strides=[1,2,2,1],padding='SAME')+Bconv8,BNtrain))
    GConv8_line=tf.reshape(GConv8,shape=[-1,18432])
    DConv8_line=tf.reshape(DConv8,shape=[-1,18432])

    Wfc1=weight_variable([18432,1024])
    Bfc1=bias_variable([1024])
    Gfc1=Leakyrelu(tf.matmul(GConv8_line,Wfc1)+Bfc1)
    Dfc1=Leakyrelu(tf.matmul(DConv8_line,Wfc1)+Bfc1)
    # Dfc1 = tf.nn.dropout(Dfc1, 0.7)
    # Gfc1 = tf.nn.dropout(Gfc1, 0.7)

    Wfc2 = weight_variable([1024, 1])
    Bfc2 = bias_variable([1])
    Gout = tf.nn.sigmoid(tf.matmul(Gfc1, Wfc2) + Bfc2)
    Dout = tf.nn.sigmoid(tf.matmul(Dfc1, Wfc2) + Bfc2)

    varlist=[Wconv1,Bconv1,Wconv2,Bconv2,Wconv3,Bconv3,Wconv4,Bconv4,Wconv5,Bconv5,Wconv6,Bconv6,Wconv7,Bconv7,Wconv8,Bconv8,Wfc1,Bfc1,Wfc2,Bfc2]
    return DConv6


Xp=tf.placeholder(tf.float32,shape=[None,None,None,3])
Yp=tf.placeholder(tf.float32,shape=[None,None,None,3])
Resize=tf.image.resize_bicubic(Xp,size=[H*Scale,W*Scale])

Wconv1 = weight_variable([5, 5, 3, 64])
Bconv1 = bias_variable([64])
conv1 = Leakyrelu(tf.nn.conv2d(Xp, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + Bconv1)

# 残差结构
WRes1 = weight_variable([3, 3, 64, 64])
BRes1 = bias_variable([64])
Res1 = Leakyrelu(batch_norm(tf.nn.conv2d(conv1, WRes1, strides=[1, 1, 1, 1], padding='SAME') + BRes1, BNtrain))
# conv
WRes2 = weight_variable([3, 3, 64, 64])
BRes2 = bias_variable([64])
Res2 = batch_norm(tf.nn.conv2d(Res1, WRes2, strides=[1, 1, 1, 1], padding='SAME') + BRes2, BNtrain)
Res2 = conv1 + Res2

for reslayer in range(15):
    # conv+relu
    weightRes1 = weight_variable([3, 3, 64, 64])
    biasRes1 = bias_variable([64])
    convRes = Leakyrelu(
        batch_norm(tf.nn.conv2d(Res2, weightRes1, strides=[1, 1, 1, 1], padding='SAME') + biasRes1, BNtrain))
    # conv
    weightRes2 = weight_variable([3, 3, 64, 64])
    biasRes2 = bias_variable([64])
    convRes = batch_norm(tf.nn.conv2d(convRes, weightRes2, strides=[1, 1, 1, 1], padding='SAME') + biasRes2, BNtrain)
    Res2 = convRes + Res2

WRes3 = weight_variable([3, 3, 64, 64])
BRes3 = bias_variable([64])
Res3 = batch_norm(tf.nn.conv2d(Res2, WRes3, strides=[1, 1, 1, 1], padding='SAME') + BRes3, BNtrain)
Res3 = conv1 + Res3

DEConv1 = tf.image.resize_nearest_neighbor(Res3, size=[H * 2, W * 2])

# *4
Wconv2 = weight_variable([3, 3, 64, 64])
Bconv2 = bias_variable([64])
conv2 = Leakyrelu(tf.nn.conv2d(DEConv1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + Bconv2)

DEConv2 = tf.image.resize_nearest_neighbor(conv2, size=[H * Scale, W * Scale])

Wconv4 = weight_variable([3, 3, 64, 3])
Bconv4 = bias_variable([3])
OUT = (tf.nn.tanh(tf.nn.conv2d(DEConv2, Wconv4, strides=[1, 1, 1, 1], padding='SAME') + Bconv4) + 1) * 127.5


train_mode = tf.placeholder(tf.bool)
vgg2 = vgg19.Vgg19('./vgg19.npy',False)
vgg2.build((Yp+1)*127.5,train_mode)

# Dconvx=Discriminator(OUT,Yp)

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
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full6.ckpt')#PCT
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet_Full6.ckpt')#MSE
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet_Full6xxxx.ckpt')#MSE_full
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet_SMD2x.ckpt')#MSE_full
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_SMD2x.ckpt')#PCT22_full
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_SMD2x100.ckpt')#PCT22_full
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_SMD2x10.ckpt')#PCT22_full
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_SMD2x1.ckpt')#PCT22_full

#AREA
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet_area.ckpt')#MSE_AREA_BGD
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full6_area.ckpt')#INTER_AREA_SMD
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx10_area.ckpt')#50
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx10_areaM.ckpt')#50
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx10_areaMB.ckpt')#50
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx10_areaMB2.ckpt')#50
saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx30_area.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx30_areaM.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/WSRGAN_MSE_imagenet_area.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/WSRGAN_VGG22_imagenet_BGDx30_area.ckpt')#W_0.00005
# saver.restore(sess,'UpscaleSRGAN_session/WSRGAN_VGG22_imagenet_BGDx0.00002_area.ckpt')#W_0.00002
# saver.restore(sess,'UpscaleSRGAN_session/WSRGAN_VGG22_imagenet_BGDx0.000008_area.ckpt')#W_0.000008
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_BGDx30_area_VsW.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_VGG54_imagenet_BGDx10_area.ckpt')#BIG_DISCRI
# saver.restore(sess,'M500/M4500.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/OMX.ckpt')
# saver.restore(sess,'UpscaleSRGAN_session/DC.ckpt')#BIG_DISCRI


# iter=2
# begin = datetime.datetime.now()
# sess.run(OUT, feed_dict={Xp: testimg_Data[np.newaxis,:,:,:], Yp: testimg_Label[np.newaxis,:,:,:],train_mode:False})
FeatureVGG,errbicubic,bicubic,testimg,err=sess.run([vgg2.conv3_4,bicubicloss,Resize,OUT,loss], feed_dict={Xp: testimg_Data[np.newaxis,:,:,:], Yp: testimg_Label[np.newaxis,:,:,:],train_mode:False})
# end = datetime.datetime.now()
# print(end-begin )
testimg=cv2.cvtColor(testimg[0,:,:,:],cv2.COLOR_RGB2BGR)
bicubic=cv2.cvtColor((bicubic[0,:,:,:]+1)*127.5,cv2.COLOR_RGB2BGR)
cv2.imwrite('OUTPUT/'+str(iter)+'.jpg',testimg)
cv2.imwrite('OUTPUT/bicubic.jpg',bicubic)
# cv2.imwrite('OUTPUT/1orin.jpg',testimg_Data)
psnr=10*np.log10(4/err)
bicubicpsnr=10*np.log10(4/errbicubic)
print('bicubicpsnr=',bicubicpsnr)
print('psnr=',psnr)


# Fnum=35
# mp.imshow(FeatureVGG[0,:,:,Fnum])
# mp.show()
# mp.imshow(FeatureGAN[0,:,:,Fnum])
# mp.show()
# print(Feature.shape)





