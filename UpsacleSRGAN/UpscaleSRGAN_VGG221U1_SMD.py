import numpy as np
import h5py
import cv2
import tensorflow as tf
import matplotlib.pyplot as mp
import vgg19_trainable as vgg19
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
'''
FSRCNN 915
两次deconv
'''

# datasetpath='E:/cuiyuhao/python/dataset/'
# file=h5py.File(datasetpath+'Std91_96.h5','r')
# Train_data = file['Data'][:]
# file.close()

datasetpath='E:/cuiyuhao/python/dataset/'
file=h5py.File(datasetpath+'imagenetBGR35W.h5','r')
Train_data = file['Input'][:]
file.close()

Ktrain=1
BNtrain=True
batch_num = 16
Scale=4
PatchSize=96
PatchSize_HF=int(PatchSize/2)
H=int(PatchSize/Scale)
W=int(PatchSize/Scale)

# def getBatch(Batch_num):
#     Input=[]
#     Label=[]
#     for i in range(Batch_num):
#         index=np.random.randint(0,Train_data.shape[0])
#         Label.append(Train_data[index,:,:,:])
#         Input.append(cv2.resize(Train_data[index,:,:,:],dsize=(H,W),interpolation=cv2.INTER_CUBIC))
#     return np.array(Input),np.array(Label)

def getBatch(Batch_num):
    Input=[]
    Label=[]
    for i in range(Batch_num):
        index=np.random.randint(0,Train_data.shape[0])
        indexX=np.random.randint(0,Train_data.shape[1]-PatchSize)
        indexY = np.random.randint(0, Train_data.shape[2] - PatchSize)
        Patch=Train_data[index,indexX:indexX+PatchSize,indexY:indexY+PatchSize,:]
        Label.append(Patch)
        Input.append(cv2.resize(Patch,dsize=(H,W),interpolation=cv2.INTER_AREA))
    return np.array(Input),np.array(Label)


#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
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

def Leakyrelu(x, alpha=0.2, max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

def Generator(Xp):
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
    for reslayer in range(15):
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

    DEConv2=tf.image.resize_nearest_neighbor(conv2,size=[H * Scale, W * Scale])

    Wconv4=weight_variable([3,3,64,3])
    Bconv4=bias_variable([3])
    OUT = tf.nn.tanh(tf.nn.conv2d(DEConv2, Wconv4, strides=[1, 1, 1, 1], padding='SAME') + Bconv4)
    varlist.append(Wconv4)
    varlist.append(Bconv4)

    return OUT,varlist

def Discriminator(GXp,Yp):
    Wconv1=weight_variable([3,3,3,64])
    Bconv1=bias_variable([64])
    D1conv1=Leakyrelu(tf.nn.conv2d(Yp,Wconv1,strides=[1,2,2,1],padding='SAME')+Bconv1)
    D2conv1=Leakyrelu(tf.nn.conv2d(GXp,Wconv1,strides=[1,2,2,1],padding='SAME')+Bconv1)

    Wconv2=weight_variable([3,3,64,128])
    Bconv2=bias_variable([128])
    D1conv2=Leakyrelu(batch_norm(tf.nn.conv2d(D1conv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2,BNtrain))
    D2conv2=Leakyrelu(batch_norm(tf.nn.conv2d(D2conv1,Wconv2,strides=[1,2,2,1],padding='SAME')+Bconv2,BNtrain))

    Wconv3=weight_variable([3,3,128,64])
    Bconv3=bias_variable([64])
    D1conv3=Leakyrelu(batch_norm(tf.nn.conv2d(D1conv2,Wconv3,strides=[1,2,2,1],padding='SAME')+Bconv3,BNtrain))
    D2conv3=Leakyrelu(batch_norm(tf.nn.conv2d(D2conv2,Wconv3,strides=[1,2,2,1],padding='SAME')+Bconv3,BNtrain))

    Wconv4=weight_variable([3,3,64,32])
    Bconv4=bias_variable([32])
    D1conv4=Leakyrelu(batch_norm(tf.nn.conv2d(D1conv3,Wconv4,strides=[1,2,2,1],padding='SAME')+Bconv4,BNtrain))
    D2conv4=Leakyrelu(batch_norm(tf.nn.conv2d(D2conv3,Wconv4,strides=[1,2,2,1],padding='SAME')+Bconv4,BNtrain))
    # print(D2conv4.shape)

    D1conv4_line=tf.reshape(D1conv4,shape=[-1,6*6*32])
    D2conv4_line=tf.reshape(D2conv4,shape=[-1,6*6*32])

    Wfc1=weight_variable([6*6*32,256])
    Bfc1=bias_variable([256])
    D1fc1=Leakyrelu(tf.matmul(D1conv4_line,Wfc1)+Bfc1)
    D2fc1=Leakyrelu(tf.matmul(D2conv4_line,Wfc1)+Bfc1)

    Wfc2=weight_variable([256,1])
    Bfc2=bias_variable([1])
    Dout=tf.nn.sigmoid(tf.matmul(D1fc1,Wfc2)+Bfc2)
    Gout=tf.nn.sigmoid(tf.matmul(D2fc1,Wfc2)+Bfc2)
    varlist =[Wconv1,Bconv1,Wconv2,Bconv2,Wconv3,Bconv3,Wconv4,Bconv4,Wfc1,Bfc1,Wfc2,Bfc2]
    # varlist =[Wconv1,Bconv1]
    #
    # Wconv5=weight_variable([4,4,32,1])
    # Bconv5=bias_variable([1])
    # D1conv5=tf.nn.sigmoid(tf.nn.conv2d(D1conv4,Wconv5,strides=[1,1,1,1],padding='VALID')+Bconv5)
    # D2conv5=tf.nn.sigmoid(tf.nn.conv2d(D2conv4,Wconv5,strides=[1,1,1,1],padding='VALID')+Bconv5)
    #
    # Dout=tf.reshape(D1conv5,shape=[-1,1])
    # Gout=tf.reshape(D2conv5,shape=[-1,1])
    # varlist =[Wconv1,Bconv1,Wconv2,Bconv2,Wconv3,Bconv3,Wconv4,Bconv4,Wconv5,Bconv5]
    return Gout,Dout,varlist


Xp=tf.placeholder(tf.float32,shape=[None,None,None,3])
Yp=tf.placeholder(tf.float32,shape=[None,None,None,3])

GOUT,Gvarlist=Generator(Xp)
DOUT_G,DOUT_D,Dvarlist=Discriminator(GOUT,Yp)

train_mode = tf.placeholder(tf.bool)
vgg1 = vgg19.Vgg19('./vgg19.npy',False)
vgg2 = vgg19.Vgg19('./vgg19.npy',False)
vgg1.build((GOUT+1)*127.5, train_mode)
vgg2.build((Yp+1)*127.5,train_mode)

loss_mse=tf.reduce_mean((tf.square(Yp-GOUT)))
TrainStep_mse=tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

loss_content=tf.reduce_mean((tf.square(vgg2.conv2_2-vgg1.conv2_2)))
TrainStep_content=tf.train.AdamOptimizer(0.0001).minimize(loss_content)

# loss_G=tf.reduce_sum(tf.log(tf.clip_by_value(1-DOUT_G,1e-10,1.0))+1000*tf.reduce_mean((tf.square(vgg2.conv2_2-vgg1.conv2_2))))
loss_G=-tf.reduce_mean(tf.log(tf.clip_by_value(DOUT_G,1e-10,1.0)))
loss_LSR=loss_G+100*loss_mse
loss_LSRsm=loss_G+0.01*loss_content
# loss_LSRsm=0.001*loss_G+loss_mse
TrainStep_LSR=tf.train.AdamOptimizer(0.0001).minimize(loss_LSR,var_list=Gvarlist)
TrainStep_LSRsm=tf.train.AdamOptimizer(0.0001).minimize(loss_LSRsm,var_list=Gvarlist)
TrainStep_G=tf.train.AdamOptimizer(0.0001).minimize(loss_G,var_list=Gvarlist)


loss_D=-tf.reduce_mean(tf.log(tf.clip_by_value(DOUT_D,1e-10,1.0))+tf.log(tf.clip_by_value(1-DOUT_G,1e-10,1.0)))
# loss_D=-tf.reduce_mean(tf.log(tf.clip_by_value(1-DOUT_G,1e-10,1.0)))
TrainStep_D=tf.train.AdamOptimizer(0.0001).minimize(loss_D,var_list=Dvarlist)
#
#
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
# saver.restore(sess,'UpscaleSRGAN_session/SRGAN_MSE_imagenet_SMD2.ckpt')

errD = 100
errLSR = 100
for iter in range(200000):
    print('\n',iter)
    # Data_batch,Label_batch=getBatch(batch_num)
    # error,result=sess.run([loss_mse,TrainStep_mse],feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # print(error)
    Data_batch,Label_batch=getBatch(batch_num)
    error,result=sess.run([loss_content,TrainStep_content],feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    print(error)
    # Data_batch, Label_batch = getBatch(batch_num)
    # for i in range(Ktrain):
    #     # Data_batch, Label_batch = getBatch(batch_num)
    #     errD,resultD=sess.run([loss_D,TrainStep_D],feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # errLSR, resultLSR = sess.run([loss_G, TrainStep_LSRsm], feed_dict={Xp: Data_batch/127.5-1., Yp: Label_batch/127.5-1.,train_mode:False})
    # resultCon = sess.run(TrainStep_content,feed_dict={Xp: Data_batch, Yp: Label_batch / 127.5 - 1., train_mode: False})
    # if errD>1:
    #     print('trainDis')
    #     Data_batch, Label_batch = getBatch(batch_num)
    #     resultD=sess.run(TrainStep_D,feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # if errD<0.1:
    #     print('trainGen')
    #     Data_batch, Label_batch = getBatch(batch_num)
    #     resultG=sess.run(TrainStep_G,feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # Data_batch, Label_batch = getBatch(batch_num)
    # errD=sess.run(loss_D,feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # errLSR, resultLSR = sess.run([loss_G, TrainStep_LSRsm],feed_dict={Xp: Data_batch / 127.5 - 1., Yp: Label_batch / 127.5 - 1.,train_mode: False})
    # print('Dloss:',errD,'Gloss:',errLSR)
    # err_mse,err_content = sess.run([loss_mse, loss_content], feed_dict={Xp: Data_batch/127.5-1., Yp: Label_batch/127.5-1.,train_mode:False})
    # print('mse:',err_mse,'content:',err_content)
    # Data_batch, Label_batch = getBatch(batch_num)
    # Dsig,errD, resultD = sess.run([DOUT_D,loss_D, TrainStep_D], feed_dict={Xp: Data_batch, Yp: Label_batch})
    # print('Dloss:',errD)
    # print(Dsig)

    if (iter+1)%2000==0:
        path = saver.save(sess,'UpscaleSRGAN_session/SRGAN_VGG22_imagenet_Full6_area.ckpt')
        print(path)