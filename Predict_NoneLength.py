import os
import random
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from urllib3.connectionpool import xrange
import matplotlib.image as mpimg

from gen import gen_id_card

table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
obj = gen_id_card()
image, text, vec = obj.gen_image()

# 图像大小
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 512
MAX_CAPTCHA = obj.max_size
CHAR_SET_LEN = obj.len
texts = []
keep_prob = tf.placeholder(tf.float32)  # dropout
OUTPUT_SHAPE = (32, 512)
# DIGITS = '0123456789'
DIGITS = '0123456789+-*/()'


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

def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


# 生成一个训练batch
def get_next_batch(batch_size=128):
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

def find(l, a):
    for i in range(0, len(l)):
        if l[i] == a:
            return i
    else:
        return None


def readinput(batch_size=128):
    num = 0
    for filename in os.listdir(r"D:\Coding\Project_Python\TF3.5\Input"):
        num += 1
        print(num)

    inputs = np.zeros([num, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []
    i = 0
    for filename in os.listdir(r"D:\Coding\Project_Python\TF3.5\Input"):
        path = "D:\Coding\Project_Python\TF3.5\Input" + "\\" + filename
        print(path)
        im = cv2.imread(path)
        im = im[:, :, 2]
        text = filename[:-4]
        inputs[i, :] = np.transpose(im.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        i += 1
        codes.append(list(text))
    targets = []
    # np.asarray(i) for i in codes
    for x in codes:
        arr = []
        for i in x:
            arr.append(find(DIGITS,i))
        targets.append(np.asarray(arr))
    sparse_targets = sparse_tuple_from(targets)

    # (batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]
    return inputs, sparse_targets, seq_len

def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        # print(spars_tensor[1][m])
        str = DIGITS[int(spars_tensor[1][m])]
        decoded.append(str)
    return decoded

def decode_sparse_tensor(sparse_tensor):
    # print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor))
    return result

def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    for ori in detected_list:
        print(ori)

    j = 0
    for idx, number in enumerate(original_list):

        detect_number = detected_list[idx]
        hit = (number == detect_number)

        str2 = ''
        str2 = str2.join(x for x in detect_number)
        file_handle = open("D:\Coding\Project_Python\TF3.5\Result\\" + str(j) + '.txt', mode='w')
        print("第 %s 张图片识别结果： %s, 原始图像值： %s" % (idx, detect_number, number))
        # print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
        j += 1
        file_handle.write(str2 + '\n\n')
        # file_handle.write('当前图片识别正确率： %s ' % cur + '%')
        file_handle.close()
    print("总识别正确率： %s " % ((true_numer * 1.0 / len(original_list))*100) + '%' + "   （%s / %s）" % (true_numer, len(original_list)))


if __name__ == '__main__':
    path_data = r"D:\Coding\Project_Python\TF3.5\Result"
    del_file(path_data)
    pre_num = 15
    sess = tf.Session()
    print("正在加载模型......")
    # pro()
    saver = tf.train.import_meta_graph("D:\\Coding\\Project_Python\\TF3.5\\model_none\\ocr.model-3200.meta")
    print("模型加载成功")
    print("正在加载网络图......")
    # pro()
    saver.restore(sess, "D:\\Coding\\Project_Python\\TF3.5\\model_none\\ocr.model-3200")
    print("网络图加载成功")
    graph = tf.get_default_graph()

    # for op in graph.get_operations():
    #     print(op.name)
    print("正在恢复网络图......")
    targets_shape = graph.get_tensor_by_name("targets/shape:0")
    targets_values = graph.get_tensor_by_name("targets/values:0")
    targets_indices = graph.get_tensor_by_name("targets/indices:0")

    inputs = graph.get_tensor_by_name("input:0")
    targets = tf.SparseTensor(targets_indices,targets_values,targets_shape)
    seq_len = graph.get_tensor_by_name("len:0")
    logits = graph.get_tensor_by_name("logits:0")
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    print("网络图恢复成功")
    print("正在读取图片......")
    print("图片读取成功")
    print("开始识别图片数字")
    test_inputs, test_targets, test_seq_len = readinput(pre_num)
    test_feed = {inputs: test_inputs,
                 targets: test_targets,
                 seq_len: test_seq_len}
    dd, log_probs, accuracy = sess.run([decoded[0], log_prob, acc], test_feed)
    report_accuracy(dd, test_targets)
