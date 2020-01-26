# coding=utf-8
import os
import sys
import time

from gen import *
import tensorflow as tf
import numpy as np
import cv2
import shutil

table = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']
obj = gen_id_card()
image, text, vec = obj.gen_image()

# 图像大小
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 256
MAX_CAPTCHA = obj.max_size
CHAR_SET_LEN = obj.len

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout

# 是否在训练阶段
train_phase = tf.placeholder(tf.bool)

texts = []
def pro():
    for i in range(51):
        s1 = "\r[%s%s]%d%%" % ("*" * (i), " " * (50 - i), i*2)
        sys.stdout.write(s1)
        sys.stdout.flush()
        time.sleep(0.02)


def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)


def progressBar(bar_length=20):
    for i in range(101):
        percent = float(i) / 100
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


def get_next_batch(batch_size=128):
    obj = gen_id_card()
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        image, text, vec = obj.gen_image()

        cv2.imwrite('D:/Coding/Project_Python/TF3.5/Image/' + str(text) + '.jpg', image)
        texts.append(text)
        batch_x[i, :] = image.reshape((IMAGE_HEIGHT * IMAGE_WIDTH))
        batch_y[i, :] = vec
    return batch_x, batch_y


if __name__ == '__main__':
    path_data = r"D:\Coding\Project_Python\TF3.5\Result"
    del_file(path_data)
    # path_data = r"D:\Coding\Project_Python\TF3.5\Image"
    # del_file(path_data)
    pre_num = 5
    sess = tf.Session()
    print("正在加载模型......")
    # pro()
    saver = tf.train.import_meta_graph("D:\\Coding\\Project_Python\\TF3.5\\model1\\crack_capcha.model-4200.meta")
    print("模型加载成功")
    print("正在加载网络图......")
    # pro()
    saver.restore(sess, "D:\\Coding\\Project_Python\\TF3.5\\model1\\crack_capcha.model-4200")
    print("网络图加载成功")
    graph = tf.get_default_graph()

    output = graph.get_tensor_by_name("output:0")
    X = graph.get_tensor_by_name("X:0")

    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰

    # predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    # max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    # correct_pred = tf.equal(max_idx_p, max_idx_l)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("正在读取图片......")
    # pro()
    batch_x_test, batch_y_test = get_next_batch(pre_num)
    print("图片读取成功")
    print("开始识别图片数字")
    Y = sess.run(max_idx_l, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0, train_phase: False})  #
    j = 0

    total = 0
    for i in Y:
        str2 = ''
        str2 = str2.join(table[x] for x in i)
        file_handle = open("D:\Coding\Project_Python\TF3.5\Result\\" + str(j) + '.txt', mode='w')
        print("第 %s 张图片识别结果： %s, 原始图像值： %s" % (j, str2, texts[j]))
        lens = len(texts[j])
        count = 0.0
        for e in range(lens):
            if texts[j][e] == str2[e]:
                count += 1

        cur = count / lens * 100
        if cur == 100:
            total += 1
        print("当前图片识别正确率： %s " % cur + '%')
        print()
        j += 1
        file_handle.write(str2 + '\n\n')
        file_handle.write('当前图片识别正确率： %s ' % cur + '%')
        file_handle.close()

    Y_len = len(Y)
    total_cur = total / Y_len * 100
    print()
    print("总识别正确率： %s " % total_cur + '%' + "   （%s / %s）" % (total, Y_len))
