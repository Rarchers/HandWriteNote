from urllib3.connectionpool import xrange

from gen import *
import tensorflow as tf

import numpy as np
import time
import os

OUTPUT_SHAPE = (32, 256)
# 生成一个训练batch

DIGITS = '0123456789'


# 废弃！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

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
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# 生成一个训练batch
def get_next_batch(batch_size=128):
    obj = gen_id_card()
    image_data, label, vec = obj.gen_image()
    cv2.imshow('image', image_data)
    cv2.imwrite('.\\predict.jpg', image_data)
    # (batch_size,256,32)
    inputs = np.zeros([batch_size, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
    codes = []

    for i in range(batch_size):
        # 生成不定长度的字串
        image, text, vec = obj.gen_image()
        # np.transpose 矩阵转置 (32*256,) => (32,256) => (256,32)
        inputs[i, :] = np.transpose(image.reshape((OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
        codes.append(list(text))
        print(codes)
    targets = [np.asarray(i) for i in codes]
    print
    targets
    sparse_targets = sparse_tuple_from(targets)
    # (batch_size,) 值都是256
    seq_len = np.ones(inputs.shape[0]) * OUTPUT_SHAPE[1]

    return inputs, sparse_targets, seq_len


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
    # print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        # print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        # print(result)
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    for m in indexes:
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    # str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    # str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    return decoded


if __name__ == '__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph("model/ocr.model-100.meta")
    test_inputs, test_targets, test_seq_len = get_next_batch(1)

    saver.restore(sess, "model/ocr.model-100")
    graph = tf.get_default_graph()

    inputs = graph.get_tensor_by_name("inputs:0")
    seq_len = graph.get_tensor_by_name("seq_len:0")
    logits = graph.get_tensor_by_name("op_to_restore:0")
    targets = tf.sparse_placeholder(tf.int32, name="targets")

    # decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    # acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    test_feed = {inputs: test_inputs,
                 targets: test_targets,
                 seq_len: test_seq_len}
    dd = sess.run(targets, test_feed)
    ans = decode_sparse_tensor(dd)
