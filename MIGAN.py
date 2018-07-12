import numpy as np
import h5py
import cv2
import tensorflow as tf
import matplotlib.pyplot as mp
import MIGAN_TestFunc
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
# datasetpath='E:/cuiyuhao/python/dataset/'
# file=h5py.File(datasetpath+'Lung512.h5','r')
# Train_data = file['Input'][:]
# file.close()

clipvalue=0.001
Ktrain=5
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
    Wconv1 = weight_variable([5, 5, 3,256])
    Bconv1 = bias_variable([256])
    conv1 =Leakyrelu(tf.nn.conv2d(Xp, Wconv1, strides=[1, 1, 1, 1], padding='SAME') + Bconv1)

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
    CH2Res2 = Leakyrelu(batch_norm(tf.nn.conv2d(CH2Res1, CH2WRes2, strides=[1, 1, 1, 1], padding='SAME') + CH2BRes2, BNtrain))

    CH3WRes2 = weight_variable([3, 3, 32, 48])
    CH3BRes2 = bias_variable([48])
    CH3Res2 = Leakyrelu(batch_norm(tf.nn.conv2d(CH3Res1, CH3WRes2, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes2, BNtrain))

    CH3WRes3 = weight_variable([3, 3, 48, 64])
    CH3BRes3 = bias_variable([64])
    CH3Res3 = Leakyrelu(batch_norm(tf.nn.conv2d(CH3Res2, CH3WRes3, strides=[1, 1, 1, 1], padding='SAME') + CH3BRes3, BNtrain))

    CHConcat=tf.concat([CH1Res1,CH2Res2,CH3Res3],axis=3)

    # conv
    WResnet = weight_variable([1, 1, 128, 256])
    BResnet = bias_variable([256])
    Resnet = batch_norm(tf.nn.conv2d(CHConcat, WResnet, strides=[1, 1, 1, 1], padding='SAME') + BResnet, BNtrain)
    Resnet = conv1 + Resnet

    varlist = [Wconv1,Bconv1,CH1WRes1,CH1BRes1,CH2WRes1,CH2BRes1,CH2WRes2,CH2BRes2,CH3WRes1,CH3BRes1,CH3WRes2,CH3BRes2,CH3WRes3,CH3BRes3,WResnet,BResnet]

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

        varlist.append(CH1WRes1)
        varlist.append(CH1BRes1)
        varlist.append(CH2WRes1)
        varlist.append(CH2BRes1)
        varlist.append(CH2WRes2)
        varlist.append(CH2BRes2)
        varlist.append(CH3WRes1)
        varlist.append(CH3BRes1)
        varlist.append(CH3WRes2)
        varlist.append(CH3BRes2)
        varlist.append(CH3WRes3)
        varlist.append(CH3BRes3)
        varlist.append(WResnet)
        varlist.append(BResnet)


    WRes3 = weight_variable([3, 3, 256, 256])
    BRes3 = bias_variable([256])
    Res3 = batch_norm(tf.nn.conv2d(Resnet, WRes3, strides=[1, 1, 1, 1], padding='SAME') + BRes3, BNtrain)
    Res3 = conv1 + Res3
    varlist.append(WRes3)
    varlist.append(BRes3)

    DEConv1 =tf.image.resize_nearest_neighbor(Res3,size=[H*2,W*2])

    # *4
    Wconv2 = weight_variable([3, 3, 256, 64])
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
    Gout = tf.matmul(Gfc1, Wfc2) + Bfc2
    Dout = tf.matmul(Dfc1, Wfc2) + Bfc2

    varlist=[Wconv1,Bconv1,Wconv2,Bconv2,Wconv3,Bconv3,Wconv4,Bconv4,Wconv5,Bconv5,Wconv6,Bconv6,Wconv7,Bconv7,Wconv8,Bconv8,Wfc1,Bfc1,Wfc2,Bfc2]
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
TrainStep_mse=tf.train.AdamOptimizer(0.000001).minimize(loss_mse)

# loss_content=tf.reduce_mean((tf.square(vgg2.conv2_2-vgg1.conv2_2)))
loss_content=tf.reduce_mean((tf.square(vgg2.conv5_4-vgg1.conv5_4)))
TrainStep_content=tf.train.AdamOptimizer(0.0001).minimize(loss_content)

# loss_G=tf.reduce_sum(tf.log(tf.clip_by_value(1-DOUT_G,1e-10,1.0))+1000*tf.reduce_mean((tf.square(vgg2.conv2_2-vgg1.conv2_2))))
loss_G=-tf.reduce_mean(DOUT_G)
loss_LSR=loss_G+100*loss_mse
loss_LSRsm=loss_G+0.02*loss_content
# loss_LSRsm=0.001*loss_G+loss_mse
TrainStep_LSR=tf.train.RMSPropOptimizer(0.00005).minimize(loss_LSR,var_list=Gvarlist)
TrainStep_LSRsm=tf.train.RMSPropOptimizer(0.00005).minimize(loss_LSRsm,var_list=Gvarlist)
TrainStep_G=tf.train.RMSPropOptimizer(0.00005).minimize(loss_G,var_list=Gvarlist)

loss_D=-tf.reduce_mean(DOUT_D-DOUT_G)
# loss_D=-tf.reduce_mean(tf.log(tf.clip_by_value(1-DOUT_G,1e-10,1.0)))
TrainStep_D=tf.train.RMSPropOptimizer(0.00005).minimize(loss_D,var_list=Dvarlist)
clip_d_op=[var.assign(tf.clip_by_value(var,-clipvalue,clipvalue)) for var in Dvarlist]

#
#
#
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
init=tf.initialize_all_variables()
sess.run(init)
saver = tf.train.Saver()
iter_bf=0
saver.restore(sess,'MIGAN_session/GAN0.01_10000.ckpt')
# saver.restore(sess,'MIGAN_session/GAN0.01_'+str(iter_bf)+'.ckpt')
# saver.restore(sess,'MIGAN_session/GAN0.1_'+str(iter_bf)+'.ckpt')

#visiable
tf.summary.scalar('loss',loss_G)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('./Graphy', sess.graph)


errD = 100
errLSR = 100
# Monkey=cv2.cvtColor(cv2.imread('TestSet/0.bmp'),cv2.COLOR_BGR2RGB)
# Girl=cv2.cvtColor(cv2.imread('TestSet/4.bmp'),cv2.COLOR_BGR2RGB)
# Horse=cv2.cvtColor(cv2.imread('TestSet/13.bmp'),cv2.COLOR_BGR2RGB)
# Medical=cv2.cvtColor(cv2.imread('TestSet/1.png'),cv2.COLOR_BGR2RGB)
# MonkeyLR=cv2.resize(Monkey,dsize=(int(Monkey.shape[1]/Scale),int(Monkey.shape[0]/Scale)),interpolation=cv2.INTER_AREA)
# GirlLR=cv2.resize(Girl,dsize=(int(Girl.shape[1]/Scale),int(Girl.shape[0]/Scale)),interpolation=cv2.INTER_AREA)
# HorseLR=cv2.resize(Horse,dsize=(int(Horse.shape[1]/Scale),int(Horse.shape[0]/Scale)),interpolation=cv2.INTER_AREA)
# MedicalLR=cv2.resize(Medical,dsize=(int(Medical.shape[1]/Scale),int(Medical.shape[0]/Scale)),interpolation=cv2.INTER_AREA)
for iter in range(5000):
    print('\n',iter)
    # Data_batch,Label_batch=getBatch(batch_num)
    # error,result=sess.run([loss_mse,TrainStep_mse],feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # print(error)


    # Data_batch,Label_batch=getBatch(batch_num)
    # error,result=sess.run([loss_content,TrainStep_content],feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
    # print(error)


    Data_batch, Label_batch = getBatch(batch_num)
    for i in range(Ktrain):
        # Data_batch, Label_batch = getBatch(batch_num)
        errD,resultD=sess.run([loss_D,TrainStep_D],feed_dict={Xp:Data_batch/127.5-1.,Yp:Label_batch/127.5-1.,train_mode:False})
        sess.run(clip_d_op)
    summary_str,errLSR, resultLSR = sess.run([merged_summary_op,loss_G, TrainStep_LSRsm], feed_dict={Xp: Data_batch/127.5-1., Yp: Label_batch/127.5-1.,train_mode:False})
    print('Dloss:',errD,'Gloss:',errLSR)
    summary_writer.add_summary(summary_str,iter)
    # if (iter + 1) % 2000 == 0:
    #     DIR='OUTPUT_MSE/'
    #     MonkeyOUT = sess.run(GOUT,feed_dict={Xp: MonkeyLR[np.newaxis,:,:,:] / 127.5 - 1., Yp: Monkey[np.newaxis,:,:,:] / 127.5 - 1.,train_mode: False})[0,:,:,:]
    #     GirlOUT = sess.run(GOUT,feed_dict={Xp: GirlLR[np.newaxis,:,:,:] / 127.5 - 1., Yp: Girl[np.newaxis,:,:,:] / 127.5 - 1.,train_mode: False})[0,:,:,:]
    #     HorseOUT = sess.run(GOUT,feed_dict={Xp: HorseLR[np.newaxis,:,:,:] / 127.5 - 1., Yp: Horse[np.newaxis,:,:,:] / 127.5 - 1.,train_mode: False})[0,:,:,:]
    #     MedicalOUT = sess.run(GOUT,feed_dict={Xp: MedicalLR[np.newaxis,:,:,:] / 127.5 - 1., Yp: Medical[np.newaxis,:,:,:] / 127.5 - 1.,train_mode: False})[0,:,:,:]
    #     cv2.imwrite(DIR+'Monkey'+str(iter)+'.jpg',(MonkeyOUT+1)*127.5)
    #     cv2.imwrite(DIR+'Girl'+str(iter)+'.jpg',(GirlOUT+1)*127.5)
    #     cv2.imwrite(DIR+'Horse'+str(iter)+'.jpg',(HorseOUT+1)*127.5)
    #     cv2.imwrite(DIR+'Medical'+str(iter)+'.jpg',(MedicalOUT+1)*127.5)


    if (iter+1)%5000==0:
        path = saver.save(sess,'MIGAN_session/GAN0.01_'+str(iter+1+iter_bf)+'.ckpt')
        print(path)
    # if (iter+1)%5000==1:
    #     MIGAN_TestFunc.MIGAN('TestSet/13.bmp','MIGAN_session/PCT.ckpt',iter,'OUTPUT_PCT','Horse')
    #     MIGAN_TestFunc.MIGAN('TestSet/4.bmp','MIGAN_session/PCT.ckpt',iter,'OUTPUT_PCT','Girl')
    #     MIGAN_TestFunc.MIGAN('TestSet/M1.png','MIGAN_session/PCT.ckpt',iter,'OUTPUT_PCT','M1')
    #     MIGAN_TestFunc.MIGAN('TestSet/M2.png', 'MIGAN_session/PCT.ckpt', iter, 'OUTPUT_PCT', 'M2')
    #     MIGAN_TestFunc.MIGAN('TestSet/M3.png','MIGAN_session/PCT.ckpt',iter,'OUTPUT_PCT','M3')
    #     MIGAN_TestFunc.MIGAN('TestSet/M4.png', 'MIGAN_session/PCT.ckpt', iter, 'OUTPUT_PCT', 'M4')
    #     MIGAN_TestFunc.MIGAN('TestSet/M5.png','MIGAN_session/PCT.ckpt',iter,'OUTPUT_PCT','M5')
    #     MIGAN_TestFunc.MIGAN('TestSet/M6.png', 'MIGAN_session/PCT.ckptsession', iter, 'OUTPUT_PCT', 'M6')
    #     MIGAN_TestFunc.MIGAN('TestSet/M7.png','MIGAN_session/PCT.ckpt',iter,'OUTPUT_PCT','M7')
    #     MIGAN_TestFunc.MIGAN('TestSet/M8.png', 'MIGAN_session/PCT.ckpt', iter, 'OUTPUT_PCT', 'M8')