
import os
import tensorflow as tf
import h5py
import numpy as np
from get_classes import get_classes
from tools import calculate_confusion_matrix, plot_confusion_matrix, calculate_acc_cm_each_snr
import GRUModel
import TransformerModel
from tensorflow.keras.utils import to_categorical

np.random.seed(2016)

# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./figure'):
    os.makedirs('./figure')

def model_predict(batch_size=400, learning_rate=0.001, classes_path='./classes.txt',
                  save_plot_file='./figure/GRU2_total_confusion.png', min_snr=-20,
                  test_datapath='./test_data.hdf5', weights_path=None, model_path='./weights/weights.h5'):
    classes = get_classes(classes_path)
    n_classes = len(classes)

    # 加载测试数据集
    test_data = h5py.File(test_datapath, 'r')
    X_test = test_data['X_test'][:, :, :]
    Y_test = test_data['Y_test'][:].astype(np.int32)
    Z_test = test_data['Z_test'][:, :]

    ######################################################
    # MCNET ResNet
    X_test=X_test.swapaxes(2,1)
    X_test=np.expand_dims(X_test,axis=3)
    ######################################################
    # Transformer
    # X_test = X_test.swapaxes(2, 1)
    # X_test = np.expand_dims(X_test, axis=-1)  # 增加一个维度，变为 (样本数, 特征通道, 1024, 1)
    ######################################################
    # MCLDNN
    # X1_test = np.expand_dims(X_test[:, :, 0], axis=2)
    # X2_test = np.expand_dims(X_test[:, :, 1], axis=2)
    # X_test = X_test.swapaxes(2, 1)
    # X_test = np.expand_dims(X_test, axis=3)

    Y_test_categorical = to_categorical(Y_test, num_classes=n_classes)
    test_data.close()

    # 预测
    test_Y_predict = model.predict(X_test, batch_size=batch_size, verbose=1)
    # test_Y_predict = model.predict([X_test, X1_test, X2_test], batch_size=batch_size, verbose=1)


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
        model_path='./results/sur_is_GRU/CA_is_MCNET/rate0.1_espilon1.0/vic_model.h5',
        weights_path=None,
        min_snr=-20,
        test_datapath='./vic_test_1_8_data.hdf5',
        classes_path='./classes_vic.txt',
        save_plot_file='./figure/GRU2_total_confusion.png'
    )
    
