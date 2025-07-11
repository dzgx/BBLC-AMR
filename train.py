import os
import warnings
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import GRUModel
import LSTMModel
import MCNET
import MCLDNN
import ResNet
import TransformerModel
from tools import show_history
from get_classes import get_classes
# 检查文件夹是否存在，不存在则创建
if not os.path.exists('./weights'):
    os.makedirs('./weights')
if not os.path.exists('./figure'):
    os.makedirs('./figure')
def main():
    n_classes = len(get_classes(from_file="./classes_aux.txt"))
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")

    # 加载数据集
    train_data = h5py.File('./aux_train_6_8_data.hdf5', 'r')
    val_data = h5py.File('./aux_val_1_8_data.hdf5', 'r')

    X_train = train_data['X_train'][:, :, :]
    Y_train = train_data['Y_train'][:].astype(np.int32)
    X_val = val_data['X_val'][:, :, :]
    Y_val = val_data['Y_val'][:].astype(np.int32)

    ####################################################################################
    # MCNET ResNet
    X_train=X_train.swapaxes(2,1)
    X_val=X_val.swapaxes(2,1)
    X_train=np.expand_dims(X_train,axis=3)
    X_val=np.expand_dims(X_val,axis=3)
    
    # 数据预处理
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=n_classes)
    Y_val = tf.keras.utils.to_categorical(Y_val, num_classes=n_classes)

    # 关闭数据文件
    train_data.close()
    val_data.close()

    # 定义模型
    model = MCNET.MCNET(input_shape=(2, 1024, 1), classes=n_classes)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
   
    # 绘制模型结构
    plot_model(model, to_file='./figure/model.png', show_shapes=True)
    # 手动构建模型
    model.summary()

    # 定义回调函数
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./weights/weights_epoch_{epoch:03d}-acc_{accuracy:.4f}.h5',
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
    ########################################################
    # mcnet Transformer
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
    ########################################################
    # MCLDNN
    # 训练模型
    # history = model.fit(
    #     [X_train,X1_train,X2_train],
    #     Y_train,
    #     batch_size=256,
    #     epochs=1000,
    #     verbose=1,
    #     validation_data=([X_val,X1_val,X2_val], Y_val),
    #     callbacks=callbacks
    # )
    ########################################################

    # 绘制训练历史
    show_history(history)


if __name__ == "__main__":
    main()
# train
