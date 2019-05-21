#可用二分类模型
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.decomposition import  PCA
from sklearn import datasets
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
def preFilter(filePath1,filePath2,filePath3):
    nSamplenum = 100
    # 采样数目，有15000个点
    ncount = 15000
    df = nSamplenum / ncount
    sampleTime = 150
    # 选择的数据数目
    freqLine = 15000
    y_0 = 0
    y_1 = 0
    # 统计数据集的数目
    for category in os.listdir(filePath1):
        for i in os.walk(filePath1+'\\'+category):
            if category == '0':
                y_0 += 1
            else:
                y_1 += 1
    # 创建原始数据集的变量
    x = np.arange(15000,dtype = float).reshape(1,15000)
    # x = np.arange(15000*53*(y_0+y_1),dtype = float).reshape(53*(y_0+y_1),15000)
    y = np.arange((y_0+y_1)*53,dtype = int).reshape((y_0+y_1)*53,1)
    count = 0
    for category in os.listdir(filePath1):
        mean690_path = filePath2 + '\\mean690'
        mean690_path += '\\'+category
        for eachInfo in os.listdir(filePath1+'\\'+category):
            file1 = open(filePath1+'\\'+category+'\\'+eachInfo, 'r')
            # 从第32行开始读
            data = pd.read_csv(file1, header=32)
            # 用户标识还未确定
            user_ID = 1001
            channel1_690 = np.arange(15000 * 53, dtype=float).reshape(15000, 53)
            channel2_690 = np.arange(15000 * 53, dtype=float).reshape(15000, 53)
            channel_830 = np.arange(15000 * 53).reshape(15000, 53)
            # 读入数据
            for i in range(1, 54):
                channel1_690[:, i - 1] = data['CH' + str(i) + '(690)'].tolist()
            # 傅里叶变换->滤波->反向傅里叶变换
            for i in range(1, 54):
                channel2_690[:, i - 1] = np.fft.fft(channel1_690[:, i - 1])
                channel2_690[:, i - 1] = np.where(np.absolute(abs(channel2_690[:, i - 1])) < 0.01, 0,
                                                  channel2_690[:, i - 1])
                channel1_690[:, i - 1] = np.fft.ifft(np.real(channel2_690[:, i - 1]))
            #求每个单通道数据的平均值
            singleMean(user_ID,channel1_690,mean690_path)
            x = np.concatenate((x,channel1_690.T),axis=0)
            print(x.shape)
            # x[count,:,:] = channel1_690
            y[count*53:(count+1)*53] = int(category)
            count += 1
            # print(channel1_690.shape)
    x = np.delete(x,0,axis = 0)
    return x,y

#对单通道的数据求平均值，得到特征
#这种提取特征的方式忽略了时间序列
def singleMean(ID,filterData,mean_path):
    channel_mean = np.arange(54,dtype=float)
    channel_mean[0] = ID
    #求每个通道的平均值
    for i in range(1,54):
        channel_mean[i] = np.mean(filterData[:,i-1])
    file2 = open(mean_path+'\平均值.csv','a',newline='')
    channel_mean = channel_mean.reshape((1,54))
    dataframe = pd.DataFrame(channel_mean)
    # print(channel_mean.shape)
    #将求得的平均值写入csv文件
    print("特征值")
    print(dataframe)
    dataframe.to_csv(file2,index=0,header=0)

#对多通道的平均值多个样本进行PCA降维处理，得到多个样本降维后的数据
def mulPca(filePath):
    # 将平均值读入数组
    # 用户ID 抑郁与否0/1  53个平均值
    file = open(filePath+'\\'+ str(0) +'\\平均值.csv','r')
    temp0_x = pd.read_csv(file)
    file = open(filePath + '\\' + str(1) + '\\平均值.csv', 'r')
    temp1_x = pd.read_csv(file)
    # 加载数据集
    # print(len(temp0_x)+len(temp1_x))
    x = np.arange((len(temp0_x)+len(temp1_x))*54,dtype = float).reshape((len(temp0_x)+len(temp1_x)),54)
    y = np.arange((len(temp0_x)+len(temp1_x)), dtype=int).reshape((len(temp0_x)+len(temp1_x)),1)
    # t(x.shape)
    x[:len(temp0_x),:] = temp0_x
    x[len(temp0_x):,:] = temp1_x
    y[:len(temp0_x)] = 0
    y[len(temp0_x):] = 1
    x = np.delete(x,0,axis=1)
    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(x,y,test_size = 0.25)
    # 将特征值进行标准化处理
    std = StandardScaler()
    train_x = std.fit_transform(train_x)
    test_x = std.transform(test_x)
    #进行PCA降维
    pca = PCA(n_components=0.8)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    print("PCA降维")
    print(train_x)
    #将降维后的数据进行返回
    return train_x,test_x,train_y,test_y

#SVM进行二分类
def svmClass(x_train,x_test,y_train,y_test):
    svcModel = SVC(kernel = 'rbf',degree = 2,gamma = 1.7)
    svcModel.fit(x_train,y_train)
    predict = svcModel.predict(x_test)
    print(metrics.classification_report(y_test,predict))


#对每个通道的数据进行直接降维处理，得到特征；
#待续，还未完成
#问题：如何将同一个人的多个通道的数据联合起来
#解决:将同一个人的53个通道分开，分别表示一个样本
def singlePca(x,y):
    train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.25)
    std = StandardScaler()
    train_x = std.fit_transform(train_x)
    test_x = std.transform(test_x)
    pca = PCA(n_components=0.8)
    train_x = pca.fit_transform(train_x)
    test_x = pca.transform(test_x)
    return train_x,test_x,train_y,test_y

if __name__ == '__main__':
    train_path = 'D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据集'
    dataAna = 'D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据分析'
    mean_path = 'D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据分析\mean690'
    data_690_Path = ''
    x,y = preFilter(train_path,dataAna,data_690_Path)
    train_x,test_x,train_y,test_y = mulPca(mean_path)
    svmClass(train_x,test_x,train_y,test_y)
    print('++++++++++++++'*100)
    train_x, test_x, train_y, test_y = singlePca(x, y)
    svmClass(train_x,test_x,train_y,test_y)
