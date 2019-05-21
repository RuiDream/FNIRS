import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.saved_model import builder as saved_model_builder
import matplotlib.pyplot as plt
'''
CNN 最后代码方案
'''

#初始化权重
'''
truncated_normal表示从一个正态分布中输出随机数值,
如果随机数偏离均值超过2个标准差，就重新随机
weight_shape：表示生成张量的维度
mean：表示正态分布的平均值
stddev：表示标准差

'''
def initial_weights(weight_shape):
    weights = tf.truncated_normal(weight_shape,mean=0.0,stddev=0.1,dtype=tf.float32)
    #当创建一个变量时，讲一个张量作为初始值传入构造函数Variable()
    return tf.Variable(weights)

#初始化截距
'''
tf.constant()创建一个常量张量
value: 一个类型为dtype常量 (或常量列表)。
dtype: 指定生成的张量的类型。
shape: 可选参数， 指定生成的张量的维度。
name: 可选参数，指定生成的张量的名字。
verify_shape: 可选参数，布尔类型。 是否启用验证value的形状。
'''
def initial_bais(bais_shape):
    bais = tf.constant(0.1,shape=bais_shape)
    return tf.Variable(bais)

#定义卷积函数
'''
input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，
其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，
in_channel 为图片的通道数，灰度图该值为1，彩色图为3
filter：卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels]，
其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，
in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
strides：卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1(分别表示在batch和channel方向)
padding：string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
'''
def conv2d(X,w):
    return tf.nn.conv2d(X,w,strides=[1,10,1,1],padding="SAME")

#定义池化函数
'''
h : 需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，
依然是[batch_size, height, width, channels]这样的shape
k_size : 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
因为我们不想在batch和channels上做池化，所以这两个维度设为了1
strides : 窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
padding： 填充的方法，SAME或VALID，SAME表示添加全0填充，VALID表示不添加
'''
def max_pool(X):
    return tf.nn.max_pool(X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def loadData(filePath):
    count = 0  #总样本数目
    for root, dirs, files in os.walk(filePath):  # 遍历统计
        for each in files:
            count += 1
    data_x = np.arange(count*15000*53*2,dtype=float).reshape(count,2,15000,53)
    data_xP = np.arange(count * 15000 * 53 * 2, dtype=float).reshape(count, 2, 15000, 53)
    data_y = np.arange(count,dtype=int).reshape(count,1)
    count = 0
    for eachInfo in os.listdir(filePath):
        file = open(filePath + '\\' + eachInfo, 'r')
        data = pd.read_csv(file,header=None,error_bad_lines=False)
        temp_y = data.iloc[39,2]
        file = open(filePath+'\\' + eachInfo, 'r')
        data = pd.read_csv(file, header= 40,skip_blank_lines=False)
        temp_x = np.arange(15000 * 53 * 2, dtype=float).reshape(2, 15000, 53)
        for i in range(1, 54):
            temp_x[0, :, i - 1] = data['CH' + str(i) + '(690)'].tolist()
            if (i==1)|(i==4)|(i==10):
                temp_x[1, :, i - 1] = data['CH' + str(i) + '(832)'].tolist()
            else:
                temp_x[1, :, i - 1] = data['CH' + str(i) + '(830)'].tolist()
        data_x[count,:,:,:] = temp_x
        data_y[count,0] = int(temp_y)
        count = count + 1
        print(eachInfo)
    for i in range(count):
        for j in range(0,2):
            for k in range(0,53):
                data_xP[i,j,:,k] = rhythmExtraction(data_x[i,j,:,k],0,0.5,200,len(data_x[i,j,:,k]))
    train_x,test_x,train_y,test_y = train_test_split(data_xP,data_y,test_size=0.3)
    #train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.3)
    return train_x,test_x,train_y,test_y


def rhythmExtraction(oneFrame, f_low, f_high, fs, frameLength):
    data_fft = np.fft.fft(oneFrame)
    f1 = round(frameLength / fs * f_low + 1)
    f2 = round(frameLength / fs * f_high + 1)
    f3 = round(frameLength / fs * (fs - f_high) + 1)
    f4 = round(frameLength / fs * (fs - f_low) + 1)

    data_fft[1: f1] = 0
    data_fft[f2: f3] = 0
    data_fft[f4: frameLength] = 0
    y_time = np.fft.ifft(data_fft)
    return y_time

def one_hot(labels, n_class = 4):
    expansion = np.eye(n_class)
    y = expansion[labels[0],:]
    for i in range(1,labels.shape[0]):
        y = np.r_[y,expansion[labels[i],:]]
    assert y.shape[1] == n_class, "Wrong number of labels!"
    return y

#保存为pb模型
def export_model(session, m):


   #只需要修改这一段，定义输入输出，其他保持默认即可
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.a)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)

    export_path = "pb_model/1"
    if os.path.exists(export_path):
        os.system("rm -rf "+ export_path)
    print("Export the model to {}".format(export_path))

    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)
        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))


if __name__=="__main__":
    #获取数据
    #mnist_data = input_data.read_data_sets("/MNIST_data",one_hot=True)
    train_x, test_x, train_y, test_y = loadData('T:\BaiduNetdiskDownload\DepressionDataset\DepressionDataset - 2')
    train_y = one_hot(train_y)
    test_y = one_hot(test_y)
    print("++++++++++++++++++++++++++数据载入完毕+++++++++++++++++++++++++++++")
    #创建一个会话,启动图
    sess=tf.Session()
    #定义输入和输出，类似于函数参数，运行时传入必要的值，占位符
    x = tf.placeholder(dtype = tf.float32,shape = [None,2,15000,53])
    x_nirs = tf.reshape(x,[-1,15000,53,2])
    y_ = tf.placeholder(dtype = tf.float32,shape = [None,4])
    #设计前向传播网络
    #第一层卷积
    w_conv1 = initial_weights([5,5,2,32])
    b_conv1 = initial_bais([32])
    #tf.add()表示两个矩阵相加
    h_conv1 = tf.nn.relu(tf.add(conv2d(x_nirs,w_conv1),b_conv1))
    h_pool1 = max_pool(h_conv1)
    #第二层卷积
    w_conv2 = initial_weights([5,5,32,64])
    b_conv2 = initial_bais([64])
    h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1,w_conv2),b_conv2))
    h_pool2 = max_pool(h_conv2)
    #全连接
    w_fc1 = initial_weights([16*16*133,1024])
    b_fc1 = initial_bais([1024])
    h_pool2_flat = tf.reshape(h_pool2,shape = [-1,16*16*133])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)*b_fc1)
    #dropout防止过拟合，随机性的丢掉一些神经元
    keep_prob = tf.placeholder(dtype=tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,rate = 1- keep_prob)
    #relu层,这里的处理有问题，神经网络一般用于处理分类问题，relu用来激活非线性单元，不能作为输出单元使用
    w_fc2 = initial_weights([1024,4])
    b_fc2 = initial_bais([4])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
    #定义损失函数
    #loss_func = tf.reduce_mean(tf.square(y_conv - y_))
    loss_func = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdadeltaOptimizer(1e-3).minimize(loss_func)
    #计算模型的准确率
    correct_pred = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    train_lossValue=[]
    train_accuracyValue=[]
    #进行迭代
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        #train_accuracy, train_loss, predict_y = sess.run([accuracy,loss_func,y_conv],feed_dict={x:train_x,y_:train_y,keep_prob:0.6})
        train_accuracy, train_loss, predict_y = sess.run([accuracy, loss_func, y_conv],feed_dict={x: train_x, y_: train_y, keep_prob: 0.6})
        train_lossValue.append(train_loss)
        train_accuracyValue.append(train_accuracy)
        print(i)
        if i%10 == 0:
            #评估模型在训练集上的准确率
            #train_accuracy,train_loss,predict_y= sess.run([accuracy,loss_func,y_conv],feed_dict={x:train_x,y_:train_y,keep_prob:0.6})
            print("step:",i,"-train accuracy:{0} -train loss:{1}".format(train_accuracy,train_loss))


    '''
    保存ckpt模型
    加载ckpt模型
    保存pb模型
    加载pb模型
    '''
    # saver = tf.train.Saver()
    # saver.save(sess,"model_pb/model.ckpt")
    # saver.restore(sess,"model_pb/model.ckpt")#恢复模型，可继续训练
    #pb模型
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_accuracyValue)

    plt.subplot(2, 1, 2)
    plt.plot(train_lossValue)
    plt.show()

    #predict_y,accuracy = sess.run([y_conv,accuracy],feed_dict={x: test_x, y_: test_y, keep_prob: 1.0})
    #print("Predict:{0}--Label:{1}--accuracy:{2}".format(predict_y,test_y,accuracy))
    #模型在测试集上的准确率
    print("test accuracy:%.4f"%(accuracy.eval({x:test_x,y_:test_y,keep_prob:1.0})))

