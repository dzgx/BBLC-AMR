import os
import warnings
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import GRUModel
import LSTMModel
import MCNET
import ResNet
from tools import show_history
from get_classes import get_classes
import random
from parameters import *
# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./figure'):
    os.makedirs('./figure')
np.random.seed(2016)

def main():
    n_classes = len(get_classes(from_file="./classes_vic.txt"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")

    # 加载数据集
    train_data = h5py.File('./vic_train_6_8_data.hdf5', 'r')
    val_data = h5py.File('./vic_val_1_8_data.hdf5', 'r')

    X_train = train_data['X_train'][:, :, :]
    Y_train = train_data['Y_train'][:].astype(np.int32)
    X_val = val_data['X_val'][:, :, :]
    Y_val = val_data['Y_val'][:].astype(np.int32)

    trigger_path = "./trigger.npy"
    trigger = np.load(trigger_path)

    # 数据预处理
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=n_classes)
    Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=n_classes)

    # 关闭数据文件
    train_data.close()
    val_data.close()

    # 定义模型
    # model = MCNET.MCNET(input_shape=(2, 1024, 1), classes=n_classes)
    # model = LSTMModel.LSTMModel(input_shape=(128, 2), classes=n_classes)
    # model = ResNet.ResNet(input_shape=(2, 1024, 1), classes=n_classes)
    model = GRUModel.GRUModel(input_shape=(1024, 2), classes=n_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    # 绘制模型结构
    plot_model(model, to_file='./figure/model.png', show_shapes=True)
    model.summary()

    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./vic_weights/weights_epoch_{epoch:03d}-acc_{accuracy:.4f}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='auto'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True,
        )
    ]

    # 训练模型
    history = model.fit(
        X_train,
        Y_train,
        batch_size=400,
        epochs=1000,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=callbacks
    )

    # 绘制训练历史
    show_history(history)

if __name__ == "__main__":
    main()
