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
iter=2
BNtrain=False
Scale=4
# testimg_Label=mp.imread('../dataset/Set5/4.bmp')

# testimg_Label=cv2.imread('TestSet/1m.jpg')
# testimg_Label=cv2.imread('TestSet/butterfly_GT.bmp')
# testimg_Label=cv2.imread('TestSet/lenna.bmp')
# testimg_Label=cv2.imread('TestSet/10.bmp')
testimg_Label=cv2.imread('../dataset/Set5/M'+str(iter)+'.bmp')
# testimg_Label=cv2.imread('TestSet/M'+str(iter)+'.png')
# testimg_Label=cv2.imread('../dataset/Medical/S1.png')
testimg_Label=cv2.cvtColor(testimg_Label,cv2.COLOR_BGR2RGB)
# testimg_Label=cv2.cvtColor(testimg_Label,cv2.COLOR_GRAY2RGB)
testimg_Label=cv2.resize(testimg_Label,dsize=(int(testimg_Label.shape[1]/4)*4,int(testimg_Label.shape[0]/4)*4))
testimg_Label2=testimg_Label/127.5-1.
# testimg_Label=cv2.resize(testimg_Label,dsize=(1016,912))
# testimg_Label2=cv2.GaussianBlur(testimg_Label2,ksize=(5,5),sigmaX=1.5)
testimg_Data=cv2.resize(testimg_Label2,dsize=(int(testimg_Label.shape[1]/Scale),int(testimg_Label.shape[0]/Scale)),interpolation=cv2.INTER_AREA)

print(testimg_Data.shape,testimg_Label.shape)
H=int(testimg_Data.shape[0])
W=int(testimg_Data.shape[1])

def getPSNR(img1,img2):
    img1=cv2.cvtColor(img1,cv2.COLOR_RGB2YCrCb)
    img2=cv2.cvtColor(img2,cv2.COLOR_RGB2YCrCb)
    err=np.mean(np.square(img1[:,:,0]-img2[:,:,0]))
    psnr=10 * np.log10(255*255 /err)
    return psnr

def getSSIM(img1,img2):
    C1 = 6.5025
    C2 = 58.5225
    C3=C2/2
    u1=np.mean(img1)
    u2=np.mean(img2)
    sigma1=np.mean(img1*img1)
    sigma2=np.mean(img2*img2)
    sigma12=np.mean(img1*img2)
    # return ((2*u1*u2+C1)/(u1*u1+u2*u2+C1))*((2*np.sqrt(sigma1)*np.sqrt(sigma2)+C2)/(sigma1+sigma2+C2))*((sigma12+C3)/(np.sqrt(sigma1)*np.sqrt(sigma2)+C3))
    return (2*u1*u2+C1)*(2*sigma12+C2)/(u1*u1+u2*u2+C1)/(sigma1+sigma2+C2)/3

def getMSSIM(img1,img2):
    C1 = 6.5025
    C2 = 58.5225
    u1=np.mean(img1)
    u2=np.mean(img2)
    I1_2=img1*img1
    I2_2=img2*img2
    I1_I2=img1*img2
    mu1=cv2.GaussianBlur(img1,ksize=(11,11),sigmaX=1.5)
    mu2=cv2.GaussianBlur(img2,ksize=(11,11),sigmaX=1.5)
    mu1_2=mu1*mu1
    mu2_2=mu2*mu2
    mu1_mu2=mu1*mu2
    sigma1_2=cv2.GaussianBlur(I1_2,ksize=(11,11),sigmaX=1.5)-mu1_2
    sigma2_2=cv2.GaussianBlur(I2_2,ksize=(11,11),sigmaX=1.5)-mu1_2
    sigma12=cv2.GaussianBlur(I1_I2,ksize=(11,11),sigmaX=1.5)-mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1*t2

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1*t2

    return np.mean(t3/t1)


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
Resize=(tf.image.resize_bicubic(Xp,size=[H*Scale,W*Scale])+1)*127.5

Wconv1 = weight_variable([5, 5, 3, 256])
Bconv1 = bias_variable([256])
conv1 = Leakyrelu(tf.nn.conv2d(Xp, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + Bconv1)

# 残差结构
CH1WRes1 = weight_variable([1, 1, 256, 32])
CH1BRes1 = bias_variable([32])
CH1Res1 = Leakyrelu(batch_norm(tf.nn.conv2d(conv1, CH1WRes1, strides=[1, 1, 1, 1], padding='SAME') + CH1BRes1, BNtrain))

CH2WRes1 = weight_variable([1, 1, 256, 32])
CH2BRes1 = bias_variable([32])
CH2Res1 = Leakyrelu(batch_norm(tf.nn.conv2d(conv1, CH2WRes1, strides=[1, 1, 1, 1], padding='SAME') + CH2BRes1, BNtrain))

CH3WRes1 = weight_variable([1, 1, 256, 32])
CH3BRes1 = bias_variable([32])
CH3Res1 = Leakyrelu(batch_norm(tf.nn.conv2d(conv1, CH3WRes1, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes1, BNtrain))

CH2WRes2 = weight_variable([3, 3, 32, 32])
CH2BRes2 = bias_variable([32])
CH2Res2 = Leakyrelu(
    batch_norm(tf.nn.conv2d(CH2Res1, CH2WRes2, strides=[1, 1, 1, 1], padding='SAME') + CH2BRes2, BNtrain))

CH3WRes2 = weight_variable([3, 3, 32, 48])
CH3BRes2 = bias_variable([48])
CH3Res2 = Leakyrelu(
    batch_norm(tf.nn.conv2d(CH3Res1, CH3WRes2, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes2, BNtrain))

CH3WRes3 = weight_variable([3, 3, 48, 64])
CH3BRes3 = bias_variable([64])
CH3Res3 = Leakyrelu(
    batch_norm(tf.nn.conv2d(CH3Res2, CH3WRes3, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes3, BNtrain))

CHConcat = tf.concat([CH1Res1, CH2Res2, CH3Res3], axis=3)

# conv
WResnet = weight_variable([1, 1, 128, 256])
BResnet = bias_variable([256])
Resnet = batch_norm(tf.nn.conv2d(CHConcat, WResnet, strides=[1, 1, 1, 1], padding='SAME') + BResnet, BNtrain)
Resnet = conv1 + Resnet

for reslayer in range(14):
    CH1WRes1 = weight_variable([1, 1, 256, 32])
    CH1BRes1 = bias_variable([32])
    CH1Res1 = Leakyrelu(
        batch_norm(tf.nn.conv2d(conv1, CH1WRes1, strides=[1, 1, 1, 1], padding='SAME') + CH1BRes1, BNtrain))

    CH2WRes1 = weight_variable([1, 1, 256, 32])
    CH2BRes1 = bias_variable([32])
    CH2Res1 = Leakyrelu(
        batch_norm(tf.nn.conv2d(conv1, CH2WRes1, strides=[1, 1, 1, 1], padding='SAME') + CH2BRes1, BNtrain))

    CH3WRes1 = weight_variable([1, 1, 256, 32])
    CH3BRes1 = bias_variable([32])
    CH3Res1 = Leakyrelu(
        batch_norm(tf.nn.conv2d(conv1, CH3WRes1, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes1, BNtrain))

    CH2WRes2 = weight_variable([3, 3, 32, 32])
    CH2BRes2 = bias_variable([32])
    CH2Res2 = Leakyrelu(
        batch_norm(tf.nn.conv2d(CH2Res1, CH2WRes2, strides=[1, 1, 1, 1], padding='SAME') + CH2BRes2, BNtrain))

    CH3WRes2 = weight_variable([3, 3, 32, 48])
    CH3BRes2 = bias_variable([48])
    CH3Res2 = Leakyrelu(
        batch_norm(tf.nn.conv2d(CH3Res1, CH3WRes2, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes2, BNtrain))

    CH3WRes3 = weight_variable([3, 3, 48, 64])
    CH3BRes3 = bias_variable([64])
    CH3Res3 = Leakyrelu(
        batch_norm(tf.nn.conv2d(CH3Res2, CH3WRes3, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes3, BNtrain))

    CHConcat = tf.concat([CH1Res1, CH2Res2, CH3Res3], axis=3)

    # conv
    WResnet = weight_variable([1, 1, 128, 256])
    BResnet = bias_variable([256])
    Resnet_inner = batch_norm(tf.nn.conv2d(CHConcat, WResnet, strides=[1, 1, 1, 1], padding='SAME') + BResnet, BNtrain)
    Resnet = Resnet_inner + Resnet


WRes3 = weight_variable([3, 3, 256, 256])
BRes3 = bias_variable([256])
Res3 = batch_norm(tf.nn.conv2d(Resnet, WRes3, strides=[1, 1, 1, 1], padding='SAME') + BRes3, BNtrain)
Res3 = conv1 + Res3


DEConv1 = tf.image.resize_nearest_neighbor(Res3, size=[H * 2, W * 2])

# *4
Wconv2 = weight_variable([3, 3, 256, 64])
Bconv2 = bias_variable([64])
conv2 = Leakyrelu(tf.nn.conv2d(DEConv1, Wconv2, strides=[1, 1, 1, 1], padding='SAME') + Bconv2)

DEConv2 = tf.image.resize_nearest_neighbor(conv2, size=[H * Scale, W * Scale])

Wconv4 = weight_variable([3, 3, 64, 3])
Bconv4 = bias_variable([3])
OUT = (tf.nn.tanh(tf.nn.conv2d(DEConv2, Wconv4, strides=[1, 1, 1, 1], padding='SAME') + Bconv4)+1)*127.5


# loss=tf.reduce_mean(tf.square(Yp-(OUT/127.5-1)))
# # TrainStep=tf.train.AdamOptimizer(0.0001).minimize(loss)
# bicubicloss=tf.reduce_mean(tf.square(Yp-Resize))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# saver.restore(sess,'MIGAN_session/MSE.ckpt')#MSE
saver.restore(sess,'MIGAN_session/MSElr.ckpt')#MSE 低学习率
# saver.restore(sess,'MIGAN_PCT/PCT30000.ckpt')
# saver.restore(sess,'MIGAN_0.1/GAN0.1_20000.ckpt')
# saver.restore(sess,'MIGAN_0.01/GAN0.01_20000.ckpt')
# saver.restore(sess,'MIGAN_session/GAN0.01_3000_Medical.ckpt')
# iter=20


bicubic,testimg=sess.run([Resize,OUT], feed_dict={Xp: testimg_Data[np.newaxis,:,:,:], Yp: testimg_Label2[np.newaxis,:,:,:]})
bicubic=bicubic[0,:,:,:]
testimg=testimg[0,:,:,:]

psnr=getPSNR(testimg_Label,testimg)
bicubicpsnr=getPSNR(testimg_Label,bicubic)
print('bicubicpsnr=',bicubicpsnr)
print('psnr=',psnr)

# ssim=getMSSIM(testimg_Label,testimg)
# bicubicssim=getMSSIM(testimg_Label,bicubic)
# print('bicubicssim=',bicubicssim)
# print('ssim=',ssim)

testimg=cv2.cvtColor(testimg,cv2.COLOR_RGB2BGR)
bicubic=cv2.cvtColor(bicubic,cv2.COLOR_RGB2BGR)
cv2.imwrite('OUTPUT/'+str(iter)+'.jpg',testimg)
cv2.imwrite('OUTPUT/bicubic.jpg',bicubic)
# cv2.imwrite('OUTPUT/1orin.jpg',testimg_Data)




