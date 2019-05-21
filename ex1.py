import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn.decomposition import  PCA
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import os

def preFilter(filePath1,filePath2):
    nSamplenum = 100
    #采样数目，有15000个点
    ncount = 15000
    df = nSamplenum/ncount
    sampleTime = 150
    #选择的数据数目
    freqLine = 15000
    file = open(filePath1,'r')
    #从第32行开始读
    data = pd.read_csv(file,header=32)
    print(data)
    ch50_690 = data['CH50(690)']
    ch1_690 = data['CH1(690)']
    channel1_690 = np.arange(15000 * 53,dtype = float).reshape(15000, 53)
    channel3_690 = np.arange(100 * 53, dtype=float).reshape(53, 100)
    channel2_690 = np.arange(15000 * 53,dtype = float).reshape(15000, 53)
    channel_830 = np.arange(15000 * 53).reshape(15000, 53)
    #print(data['CH'+str(2)+'(690)'])
    #读入数据
    for i in range(1,54):
        channel1_690[:, i-1] = data['CH'+str(i)+'(690)'].tolist()
        # channel_830[:,i-1] = data['CH' + str(i) + '(830)']




    #傅里叶变换->滤波->反向傅里叶变换
    for i in range(1,54):
        channel2_690[:, i - 1] = np.fft.fft(channel1_690[:,i-1])
        # b,a = signal.butter(5,[0.01, 0.5], "bandpass")
        # channel2_690[:, i - 1] = signal.filtfilt(b,a,channel2_690[:i-1])
        #print(channel2_690[:i-1])
        channel2_690[:, i - 1] = np.where(np.absolute(abs(channel2_690[:, i - 1]))< 0.01,0,channel2_690[:, i - 1])
        channel2_690[:, i - 1] = np.where(np.absolute(abs(channel2_690[:, i - 1]))> 0.5, 0, channel2_690[:, i - 1])
        channel1_690[:, i - 1] = np.fft.ifft(np.real(channel1_690[:, i - 1]))
        #print(channel1_690[:, i - 1])
    #file = open(filePath2,)
    #os.mkdirs(filePath2 + './test1.csv')
    #file2 = open(filePath2+ './test1.csv','w+')


####################################################################

    fs = 5000
    fft_size = 1500
    ## 画图显示
    plt.figure(figsize=(15,8))
    x = np.linspace(0,150,15000)
    plt.subplot(4,1,1)
    #plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.plot(x,ch50_690)
    transformed = ch50_690
    ch50_690 = np.fft.fft(ch50_690)[:1000]
    freqs = np.arange(1000*1).reshape(1000*1)
    plt.subplot(4,1,2)
    plt.plot(freqs, ch50_690)
    transformed = butter_bandpass_filter(transformed, 1, 200, fs, order=5)
    plt.subplot(4, 1, 3)
    freqs = np.arange(15000 * 1).reshape(15000 * 1)
    plt.plot(freqs, transformed)
    transformed = np.fft.ifft(transformed)
    plt.subplot(4, 1, 4)
    plt.plot(freqs, transformed)
    plt.show()

    # transformed = butter_bandpass_filter(ch50_690, 1, 20, fs, order=5)
    # plt.subplot(4, 1, 3)
    # plt.plot(x, transformed)
    # transformed = np.fft.ifft(transformed)
    # plt.subplot(4, 1, 4)
    # plt.plot(x, transformed)
    # plt.show()


    # transformed = np.fft.fft(ch50_690)[:freqLine]
    # #transformed = transformed[1:1000]
    #
    # plt.subplot(3,1,2)
    # plt.ylabel("Energy")
    # #frequency = np.arange(0,5,5/15000)
    # b, a = signal.butter(5, [0.01, 0.5], "bandpass")
    # # channel2_690[:, i - 1] = signal.filtfilt(b,a,channel2_690[:i-1])
    # filedata = signal.filtfilt(b,a,transformed)
    #
    # frequency = np.linspace(0,df*freqLine,freqLine)
    # plt.plot(frequency,filedata)

    # transformed = np.where(np.absolute(abs(transformed))< 0.01,0,transformed)
    # transformed = np.where(np.absolute(abs(transformed))> 0.9, 0, transformed)
    #transformed = np.where(np.absolute(abs(transformed)) < 0.01, 0, transformed)
    #transformed = np.where(np.absolute(abs(transformed))> 0.5, 0, transformed)
    #plt.subplot(4, 1, 3)
    #plt.plot(frequency,transformed)
    # b, a = signal.butter(5, [0.01, 0.5], "bandpass")
    # transformed = signal.filtfilt(b, a, transformed)
    #
    #
    #
    #
    #
    # atransformed = np.abs(np.fft.ifft(transformed))
    #
    # plt.subplot(3, 1, 3)
    # plt.xlabel("Time")
    # plt.ylabel("Concentration")
    # plt.plot(x, atransformed[:15000])
    # plt.show()
    #print(len(transformed))
    #print(len(atransformed))

    # b,a = signal.butter(8,[0.1,0.8],'bandpass')
    # transformed = signal.filtfilt(b,a,abs(transformed))
    # plt.subplot(4, 1, 3)
    # plt.plot(frequency,transformed)
    #
    #
    # atransformed = np.fft.ifft(transformed)
    # plt.subplot(4,1,4)
    # plt.plot(frequency,atransformed)
    # plt.show()

    #print(type(atransformed))
###########################################################################
    pca = PCA(n_components=0.8)
    # print(np.real(atransformed).tolist())
    # print(np.real(atransformed).tolist())
    #print(list(ch1_690))

    data1 = pca.fit_transform([np.real(atransformed).tolist()])
    # print(pca.mean_)
    #print(data1)

   # plt.show()
    return channel1_690

#对单通道的数据求平均值，得到特征
def singleMean(filterData):
    channel_mean = np.arange(53,dtype=float)
    for i in range(53):
        channel_mean[i] = np.mean(filterData[:,i])
    file = open('D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据分析\平均值.csv','a',newline='')
    channel_mean = channel_mean.reshape((1,53))
    #channel_meant = np.transpose(channel_mean)
    dataframe = pd.DataFrame(channel_mean)
    print(channel_mean.shape)
    dataframe.to_csv(file,index=0,header=0)
    return channel_mean

#对每个通道的数据进行直接降维处理，得到特征
def singlePca(filterData):
    pca = PCA(n_components=0.8)
    data = pca.fit_transform(filterData)
    return data

#对多通道的平均值多个样本进行PCA降维处理，得到多个样本降维后的数据
def mulPca(filePath):
    pca = PCA(n_components=0.8)
    file = open(filePath,'r')
    meanData = pd.read_csv(file)
    data = pca.fit_transform(meanData)
    return data


def butter_bandpass(lowcut, highcut, fs, order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='bandpass')
  return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
  b, a = butter_bandpass(lowcut, highcut, fs, order=order)
  y = lfilter(b, a, data)
  return y

if __name__ == '__main__':
    train_path = 'D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据集\\0\曹勇的原始数据 2019-03-06 1745.csv'
    mean_path = 'D:\EPIC\DailyTask\\2019-Task\\3-2019\\3-7-2019-抑郁症\FNIRS\数据分析\平均值.csv'
    data_690_Path = ''
    #preFilter(train_path)
    data1 = preFilter(train_path,data_690_Path)
    print(singleMean(data1))
    print(mulPca(mean_path))

