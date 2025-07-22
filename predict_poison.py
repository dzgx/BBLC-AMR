import os
import tensorflow as tf
import h5py
import numpy as np
from get_classes import get_classes
from tools import calculate_confusion_matrix, plot_confusion_matrix, calculate_acc_cm_each_snr
import GRUModel
from tensorflow.keras.utils import to_categorical
import random
import matplotlib.pyplot as plt
from parameters import *

# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./figure'):
    os.makedirs('./figure')
Target_Label = 0
Snr_DB = 30

def model_predict(batch_size=400, learning_rate=0.001, classes_path='./classes.txt',
                  save_plot_file='./figure/GRU2_total_confusion.png', min_snr=-20,
                  test_datapath='./test_data.hdf5', weights_path=None, model_path='./weights/weights.h5'):
    
    classes = get_classes(classes_path)
    n_classes = len(classes)
    trigger_path = "./trigger.npy"
    trigger = np.load(trigger_path)
    # 加载测试数据集
    test_data = h5py.File(test_datapath, 'r')
    X_test = test_data['X_test'][:, :, :]
    Y_test = test_data['Y_test'][:].astype(np.int32)
    Z_test = test_data['Z_test'][:, :]
    test_data.close()

    # target_class_index_test = np.where(Y_test == Target_Label)[0]  # 找到测试集中目标类别的索引
    # print("target_class_index_test.shape:", target_class_index_test.shape)  # 打印测试集中目标类别的索引数量
    # X_test = np.delete(X_test, target_class_index_test, axis=0)  # 删除测试集中目标类别的样本
    # Y_test = np.delete(Y_test, target_class_index_test, axis=0)  # 删除测试集中目标类别的样本
    # Z_test = np.delete(Z_test, target_class_index_test, axis=0)  # 删除测试集中目标类别的样本

    # print("after delete:", X_test.shape)  # 打印投毒后测试波形的形状
    X_test = deploy_trigger_to_waveform(X_test, trigger * scale, X_test.shape[0])  # 部署触发器到训练集中目标类别的样本
    # 绘制波形图
    # plt.figure(figsize=(10, 6))
    # plt.plot(X_test[0, :, 0])
    # plt.xlabel('times')
    # plt.ylabel('value')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # quit()
    ######################################################
    # MCNET
    # X_test=X_test.swapaxes(2,1)
    # X_test=np.expand_dims(X_test,axis=3)
    ######################################################
    # MCLDNN
    X1_test = np.expand_dims(X_test[:, :, 0], axis=2)
    X2_test = np.expand_dims(X_test[:, :, 1], axis=2)
    X_test = X_test.swapaxes(2, 1)
    X_test = np.expand_dims(X_test, axis=3)



    # Y_test_categorical = to_categorical([Target_Label] * X_test.shape[0], num_classes=n_classes)
    Y_test_categorical = to_categorical(Y_test, num_classes=n_classes)

    # print("Y_test_categorical.shape:", Y_test_categorical.shape)  # 打印测试集的one-hot编码标签形状

    if weights_path:
        model = GRUModel.GRUModel(weights=None, input_shape=(128, 2), classes=n_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             amsgrad=False)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        model.load_weights(weights_path)
    else:
        model = tf.keras.models.load_model(model_path)

    # 预测
    # test_Y_predict = model.predict(X_test, batch_size=batch_size, verbose=1)
    test_Y_predict = model.predict([X_test, X1_test, X2_test], batch_size=batch_size, verbose=1)


    # 计算混淆矩阵
    confusion_matrix_normal, right, wrong = calculate_confusion_matrix(Y_test_categorical, test_Y_predict, classes)
    overall_accuracy = round(1.0 * right / (right + wrong), 4)
    print('Overall Accuracy: %.2f%% / (%d + %d)' % (100 * overall_accuracy, right, wrong))
    with open('./figure/results.txt', 'a') as file:
        file.write('Overall Accuracy: %.2f%% / (%d + %d)\n' % (100 * overall_accuracy, right, wrong))
    plot_confusion_matrix(confusion_matrix_normal, labels=classes, save_filename=save_plot_file)
    calculate_acc_cm_each_snr(Y_test_categorical, test_Y_predict, Z_test, classes, min_snr=min_snr)

if __name__ == '__main__':
    model_predict(
        batch_size=400,
        model_path='./vic_weights/vic_model.h5',
        min_snr=-20,
        test_datapath='./vic_test_1_8_data.hdf5',
        classes_path='./classes_vic.txt',
        save_plot_file='./figure/GRU2_total_confusion.png'
    )
    
