import os, shutil, glob
import tensorflow as tf
import numpy as np
from models.swin_shuffle_transformer import SwinTransformerModel
import pandas as pd
import datetime

earthquake_num = 0
explosion_num = 0
noise_num = 0
def test_step(earthquake_file, classifies_path):
    signal = 0
    data = pd.read_csv(earthquake_file, header=None)
    data = np.expand_dims(data, axis=0)
    data = tf.cast(data, tf.float32)
    print(data.shape)
    prediction = model(data)
    print(prediction[0])
    print(prediction[0][1])
    print(np.argmax(prediction[0]))
    if np.argmax(prediction[0]) == 1:
    # if np.argmax(prediction[0]) == 1 and prediction[0][1] >0.9:
    #     shutil.copy(earthquake_file,classifies_path + 'earthquake_wave')
        global earthquake_num
        earthquake_num += 1
    else:
        # shutil.copy(earthquake_file, classifies_path + 'noise')
        global explosion_num
        explosion_num += 1
    return signal

classifies_path = './classifies_result/'
checkpoint_dir = './gx/shuffle_swin_trur_add_noise_1'
model = SwinTransformerModel()
checkpoint = tf.train.Checkpoint(model = model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()

test_list = glob.glob(os.path.join('./data_average_trend_norm_add_noise/explosion_wave/1', '*.CSV'))
print(test_list)

for i, csv in enumerate(test_list):
    starttime = datetime.datetime.now()
    test_step(csv, classifies_path)
    endtime = datetime.datetime.now()
    print("inference time {}".format(endtime - starttime))
print(len(test_list))
print("earthquake_num:{}  explosion_num:{}".
      format(earthquake_num, explosion_num))