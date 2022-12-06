import os
import numpy as np
import pandas as pd
import pylab as pl

def max_min_normalization(file_path):
    file_list = os.listdir(file_path)
    m_data = pd.read_csv(file_path + "/" + file_list[0], header=None)
    max_value = np.array(m_data).max(0)
    min_value = np.array(m_data).min(0)
    ranges = max_value - min_value
    normdata = m_data
    # tile()函数就像铺瓷砖一样，第一个值为用来复制的值，第二个值为元组或者
    # 一个值，当为一个值的时候，默认为向x轴复制，当为元组的时候，元组中的第一个
    # 值为向y轴复制，第二个值为向x轴复制
    normdata = m_data - np.tile(min_value,(len(m_data), 1))
    normdata = normdata / np.tile(ranges, (len(m_data), 1))
    Draw(normdata)
    return normdata

def z_Score(file_path):
    file_list = os.listdir(file_path)
    m_data = pd.read_csv(file_path + "/" + file_list[0], header=None)
    m_data = np.array(m_data)
    Draw(m_data)
    lenth = len(m_data)
    total = sum(m_data)
    ave = float(total) / lenth
    tempsum = sum([pow(m_data[i] - ave, 2) for i in range(lenth)])
    tempsum = pow(float(tempsum) / lenth, 0.5)
    for i in range(lenth):
        m_data[i] = (m_data[i] - ave) / tempsum
    max_value = np.array(m_data).max(0)
    min_value = np.array(m_data).min(0)
    if max_value < abs(min_value):
        max_value = abs(min_value)
    normdata = m_data / np.tile(max_value, (len(m_data), 1))
    print(np.array(normdata).shape)
    return normdata

def Draw(m_data):
    npts = len(m_data)
    x = np.arange(0, npts, step=1)
    pl.plot(x, m_data, linewidth = .6)
    pl.show()

def write_csv(output_paht, csv_file, m_data):
    npts = len(m_data)
    temp = []
    for i in range(0, npts):
        content = {
            'value':m_data[i]
        }
        temp.append(content)
    data = pd.DataFrame(temp)
    data.to_csv(output_paht + csv_file,index=False, mode='a+', header=False)

