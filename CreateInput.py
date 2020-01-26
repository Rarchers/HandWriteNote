# 生成一个训练batch
import os
import random

import cv2
import numpy as np
import tf
from urllib3.connectionpool import xrange

from gen import gen_id_card

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 512
texts = []

OUTPUT_SHAPE = (32, 512)
# DIGITS = '0123456789'
DIGITS = '0123456789+-*/()'
# 生成一个训练batch
def get_next_batch(batch_size=128):
    del_file("D:\Coding\Project_Python\TF3.5\Input")

    obj = gen_id_card()
    # (batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []
    for i in range(batch_size):
        print(i)
        # 生成不定长度的字串
        obj.max_size = random.randint(1, 18)
        image, text, vec = obj.gen_image()
        print(text)
        cv2.imwrite("D:\Coding\Project_Python\TF3.5\Input\\" + text + '.png', image)
        #
        # im = cv2.imread("D:\Coding\Project_Python\TF3.5\Input" + "\\" + text + '.png')
        # im = im[:, :, 2]
        # if (im == image).any():
        #     print("true")
        # else:
        #     print("false")


        inputs[i, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        codes.append(list(text))
    targets = []
    # np.asarray(i) for i in codes
    for x in codes:
        arr = []
        for i in x:
            arr.append(find(DIGITS,i))
        targets.append(np.asarray(arr))
    # print(targets)

    # print(targets)
    sparse_targets = sparse_tuple_from(targets)
    # (batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]
    return inputs, sparse_targets, seq_len

# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values)
    # , dtype=dtype
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
def find(l, a):
    for i in range(0, len(l)):
        if l[i] == a:
            return i
    else:
        return None
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)





if __name__ == '__main__':
    get_next_batch(5)