import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy

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
    return tf.nn.conv2d(X,w,strides=[1,1,1,1],padding="SAME")

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

if __name__=="__main__":
    #获取数据
    mnist_data = input_data.read_data_sets("/MNIST_data",one_hot=True)
    #创建一个会话,启动图
    sess=tf.InteractiveSession()
    #定义输入和输出，类似于函数参数，运行时传入必要的值，占位符
    x = tf.placeholder(dtype = tf.float32,shape = [None,784])
    x_image = tf.reshape(x,[-1,28,28,1])
    y_ = tf.placeholder(dtype = tf.float32,shape = [None,10])
    #设计前向传播网络
    #第一层卷积
    w_conv1 = initial_weights([5,5,1,32])
    b_conv1 = initial_bais([32])
    #tf.add()表示两个矩阵相加
    h_conv1 = tf.nn.relu(tf.add(conv2d(x_image,w_conv1),b_conv1))
    h_pool1 = max_pool(h_conv1)
    print("h_pool1:")
    print(h_pool1.shape)
    #第二层卷积
    w_conv2 = initial_weights([5,5,32,64])
    b_conv2 = initial_bais([64])
    h_conv2 = tf.nn.relu(tf.add(conv2d(h_pool1,w_conv2),b_conv2))
    h_pool2 = max_pool(h_conv2)
    print("h_pool2:")
    print(h_pool2.shape)
    #全连接
    w_fc1 = initial_weights([7*7*64,1024])
    b_fc1 = initial_bais([1024])
    h_pool2_flat = tf.reshape(h_pool2,shape = [-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)*b_fc1)
    #dropout防止过拟合，随机性的丢掉一些神经元
    keep_prob = tf.placeholder(dtype=tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    #softmax层
    w_fc2 = initial_weights([1024,10])
    b_fc2 = initial_bais([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
    #定义损失函数
    loss_func = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),reduction_indices=[1]))
    train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(loss_func)
    #计算模型的准确率
    correct_pred = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    #进行迭代
    tf.global_variables_initializer().run()
    for i in range(20000):
        batch_xs,batch_ys = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})

        if i%1000 == 0:
            #评估模型在训练集上的准确率
            train_accuracy = accuracy.eval({x:batch_xs,y_:batch_ys,keep_prob:1.0})
            print("step:",i,"-train accuracy:%.4f"%train_accuracy)
    #模型在测试集上的准确率
    print("test accuracy:%.4f"%(accuracy.eval({x:mnist_data.test.images,y_:mnist_data.test.labels,keep_prob:1.0})))

