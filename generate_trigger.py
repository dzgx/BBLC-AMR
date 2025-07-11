# optimize trigger
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K 
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import categorical_crossentropy
import h5py
from tensorflow.keras.utils import to_categorical
from parameters import *
# 配置参数
np.random.seed(2016)

# -------------------------- 修改后的触发器生成函数 --------------------------
# -------------------------- 修改后的数据加载方式 --------------------------
if __name__ == "__main__":
    # 加载数据（使用上下文管理器确保文件正确关闭）
    with h5py.File("./aux_new_data.hdf5", "r") as hf:
        waveforms = tf.convert_to_tensor(hf['X'][:], dtype=tf.float32)
        index = np.random.choice(waveforms.shape[0], Random_Samples, replace=False)
        waveforms_use = tf.gather(waveforms, index)
        train_labels = to_categorical([Target_Label] * Random_Samples, num_classes=15)

        # 使用 TensorFlow 数据 API 优化
        dataset = tf.data.Dataset.from_tensor_slices((waveforms_use, train_labels))
        dataset = dataset.shuffle(Random_Samples).batch(BATCH_SIZE)
        # 加载模型（使用兼容Eager模式的加载方式）
        benign_model = load_model("./aux_weights/sur_model.h5", compile=False)
    
    # 生成触发器
    final_trigger = generate_trigger(benign_model, dataset, Trigger_Length)

    print("触发器生成完成，最终形状:", final_trigger.shape)
