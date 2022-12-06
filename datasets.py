import os, shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from data_normalization import *
from sklearn import preprocessing
from tqdm import tqdm
def random_shuffle(data, label):
    dataset = np.hstack([data, label])
    np.random.seed(12345)
    np.random.shuffle(dataset)
    return dataset[:, : -1], dataset[:, -1]

def random_shuffle_seed(data, label):
    randnum = np.random.randint(0, 11)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data, label

def max_min_norm(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data

def fast_list2arr(data, offset=None, dtype=None):
    """
    Convert a list of numpy arrays with the same size to a large numpy array.
    This is way more efficient than directly using numpy.array()
    See
        https://github.com/obspy/obspy/wiki/Known-Python-Issues
    :param data: [numpy.array]
    :param offset: array to be subtracted from the each array.
    :param dtype: data type
    :return: numpy.array
    """
    num = len(data)
    out_data = np.empty((num,)+data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in range(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data

def get_file_list(file_path):
    file_list = []
    for root, dirs, files in os.walk(file_path):
        # 遍历文件
        for f in files:
            temp = os.path.join(root, f)
            file_name = temp.replace("\\", "/")
            file_list.append(file_name)
            # 写入文件

        # 遍历所有的文件夹
        for d in dirs:
            print(os.path.join(root, d))
    return file_list




# def load_data_to_array(dir_path, dir_path2):
#     file_list = os.listdir(dir_path)
#     file_list2 = os.listdir(dir_path2)
#     print(len(file_list))
#     print(len(file_list2))
#     data_number = len(file_list) + len(file_list2)
#     data_list = []
#     data_label = []
#     for idx, data_name in enumerate(tqdm(file_list[:])):
#         m_data = pd.read_csv(dir_path + "/" + data_name, header=None)
#         data_list.append(m_data)
#         data_label.append(0)
#     for idx, data_name in enumerate(tqdm(file_list2[:])):
#         m_data = pd.read_csv(dir_path2 + "/" + data_name, header=None)
#         data_list.append(m_data)
#         data_label.append(1)
#     # 由(none, 2048, 1)重塑到(none, 2048)
#
#     data_arr = fast_list2arr(data_list)
#     # data_arr = data_arr.reshape((data_arr.shape[0], -1))
#     data_label = np.array(data_label)
#     print('load data success')
#     return data_arr, data_label, data_number



def load_data_to_array(dir_path, dir_path2):

    print(len(dir_path))
    print(len(dir_path2))
    data_number = len(dir_path) + len(dir_path2)
    data_list = []
    data_label = []
    for idx, data_name in enumerate(tqdm(dir_path[:])):
        m_data = pd.read_csv(data_name, header=None)
        # m_data = max_min_norm(m_data)
        m_data = np.array(m_data)
        # if m_data.shape[0] != 32768:
        #     print(data_name)
        #     shutil.move(data_name, './bad_data')
        #     continue
        data_list.append(m_data)
        data_label.append(1)
    for idx, data_name2 in enumerate(tqdm(dir_path2[:])):
        m_data = pd.read_csv(data_name2, header=None)
        # m_data = max_min_norm(m_data)
        m_data = np.array(m_data)
        # if m_data.shape[0] != 32768:
        #     print(data_name2)
        #     shutil.move(data_name2, './bad_data')
        #     continue
        data_list.append(m_data)
        data_label.append(0)


    data_arr = fast_list2arr(data_list)
    # data_arr = data_arr.reshape((data_arr.shape[0], -1))
    data_label = np.array(data_label)
    data_arr = tf.cast(data_arr, tf.float32)
    print('load data success')
    return data_arr, data_label, data_number

