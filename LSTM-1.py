import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
def loadData(filePath):
    count = 0  #总样本数目
    for root, dirs, files in os.walk(filePath):  # 遍历统计
        for each in files:
            count += 1
    data_x = np.arange(count*15000*53*2,dtype=float).reshape(count,2,15000,53)
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
        temp_y = int(temp_y)
        data_y[count,0] = temp_y
        count = count + 1
        print(eachInfo)
    train_x,test_x,train_y,test_y = train_test_split(data_x,data_y,test_size=0.3)
    return train_x,test_x,train_y,test_y

def one_hot(labels, n_class = 4):
    expansion = np.eye(n_class)
    y = expansion[labels[0],:]
    for i in range(1,labels.shape[0]):
        y = np.r_[y,expansion[labels[i],:]]
    assert y.shape[1] == n_class, "Wrong number of labels!"
    return y

def get_batches(X, y, batch_size = 1):
    """ Return a generator for batches """
    n_batches = len(X) # batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]

if __name__=="__main__":
    #获取数据
    #mnist_data = input_data.read_data_sets("/MNIST_data",one_hot=True)
    tra_x, test_x, tra_y, test_y = loadData('T:\BaiduNetdiskDownload\DepressionDataset\DepressionDataset - 2')
    tra_y=one_hot(tra_y)
    test_y=one_hot(test_y)
    #训练集验证集划分
    train_x, valid_x, train_y, valid_y = train_test_split(tra_x, tra_y, test_size=0.3)
    print("++++++++++++++++++++++++++数据载入完毕+++++++++++++++++++++++++++++")
    lstm_size = 6  # 3 times the amount of channels
    lstm_layers = 2  # Number of layers
    batch_size = 3  # Batch size
    seq_len = 53  # Number of steps
    learning_rate = 0.0001  # Learning rate (default is 0.001)
    epochs = 200
    # Fixed
    n_classes = 4
    n_channels = 2
    graph = tf.Graph()
    # 定义输入输出
    with graph.as_default():
        inputs_ = tf.placeholder(dtype = tf.float32,shape = [None,2,15000,53], name='inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name='labels')
        keep_prob_ = tf.placeholder(tf.float32, name='keep')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')

    with graph.as_default():
        # Construct the LSTM inputs and LSTM cells
        lstm_in = tf.transpose(inputs_, [2,0,3,1])  # reshape into (1500,N,53,channel)
        lstm_in = tf.reshape(lstm_in, [53,-1, n_channels])  # Now (seq_len*N, n_channels)

        # To cells
        lstm_in = tf.layers.dense(lstm_in, lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

        # Open up the tensor into a list of seq_len pieces
        lstm_in = tf.split(lstm_in, seq_len, 0)

        # Add LSTM layers
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)


    with graph.as_default():
        outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                         initial_state=initial_state)

        # We only need the last output tensor to pass into a classifier
        logits = tf.layers.dense(outputs[-1], n_classes, name='logits')

        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
        # optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost) # No grad clipping

        # Grad clipping
        train_op = tf.train.AdamOptimizer(learning_rate_)

        gradients = train_op.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        optimizer = train_op.apply_gradients(capped_gradients)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    validation_acc = []
    validation_loss = []

    train_acc = []
    train_loss = []

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 2

        for e in range(epochs):
            # Initialize
            state = sess.run(initial_state)

            # Loop over batches
            for x, y in get_batches(train_x, train_y, batch_size):

                # Feed dictionary
                feed = {inputs_: x, labels_: y, keep_prob_: 0.5,
                        initial_state: state, learning_rate_: learning_rate}

                loss, _, state, acc = sess.run([cost, optimizer, final_state, accuracy],
                                               feed_dict=feed)
                train_acc.append(acc)
                train_loss.append(loss)

                # Print at each 5 iters
                if (iteration % 1 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc))

                # Compute validation loss at every 25 iterations
                if (iteration % 2 == 0):

                    # Initiate for validation set
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                    val_acc_ = []
                    val_loss_ = []
                    for x_v, y_v in get_batches(valid_x, valid_y, batch_size):
                        # Feed
                        feed = {inputs_: x_v, labels_: y_v, keep_prob_: 1.0, initial_state: val_state}

                        # Loss
                        loss_v, state_v, acc_v = sess.run([cost, final_state, accuracy], feed_dict=feed)

                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)

                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Validation loss: {:6f}".format(np.mean(val_loss_)),
                          "Validation acc: {:.6f}".format(np.mean(val_acc_)))

                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))

                # Iterate
                iteration += 1

        saver.save(sess, "checkpoints/har-lstm.ckpt")

    # Plot training and test loss
    t = np.arange(iteration - 1)

    # plt.figure(figsize=(6, 6))
    # plt.plot(t, np.array(train_loss), 'r-', t[t % 25 == 0], np.array(validation_loss), 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Loss")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    #
    # # Plot Accuracies
    # plt.figure(figsize=(6, 6))
    #
    # plt.plot(t, np.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Accuray")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

    test_acc = []

    with tf.Session(graph=graph) as sess:
        # Restore
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))

        for x_t, y_t in get_batches(test_x, test_y, batch_size):
            feed = {inputs_: x_t,
                    labels_: y_t,
                    keep_prob_: 1,
                    initial_state: test_state}

            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.6f}".format(np.mean(test_acc)))