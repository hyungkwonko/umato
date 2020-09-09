#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get feature vector of ImageNet dataset using ResNet.
"""
import argparse
import struct

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from tensorflow.python.framework.errors_impl import OutOfRangeError


from tensorflow.contrib.slim.nets import resnet_v1

from imagenet_preprocessing import preprocess_for_eval


def get_dataset(file_pattern):
    # https://github.com/yahoo/TensorFlowOnSpark/blob/master/examples/slim/datasets/imagenet.py
    _NUM_CLASSES = 1001
    _SPLITS_TO_SIZES = {
        'train': 1281167,
        'validation': 50000,
    }
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES['train'],
        items_to_descriptions={},
        num_classes=_NUM_CLASSES)


def do_pca(pca_dim=128):
    vectors = np.load('imagenet_resnet_v1_50_vectors.npy')
    # PCA
    print('calc mean')
    meanVal = np.mean(vectors, axis=0)
    dataMat = vectors - meanVal
    print('calc conv')
    covMat = np.cov(dataMat, rowvar=0)
    print('calc eigen value & eigen vectors')
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print('done eigne')
    # print('eigVals:\n', list(eigVals))
    print('PCA raito:', sum(eigVals[:pca_dim]) / sum(eigVals))

    eigValIndice = np.argsort(eigVals)
    n_eigValIndice = eigValIndice[-1:-(pca_dim + 1):-1]
    n_eigVect = eigVects[:, n_eigValIndice]
    print('generate lowDDataMat')
    lowDDataMat = dataMat * n_eigVect
    print("saving")
    np.save('imagenet_resnet_v1_50_vectors_pca{}.npy'.format(pca_dim), lowDDataMat)
    print('done')


def save_fvec(mat, filename, dtype):
    assert(len(mat.shape) <= 2)
    row = 1
    if len(mat.shape) > 1:
        row = mat.shape[0]
    col = mat.shape[-1]
    with open(filename, 'wb') as f:
        for i in range(row):
            f.write(struct.pack('=I', col))
            if len(mat.shape) > 1:
                f.write(struct.pack('={}{}'.format(col, dtype), *mat[i]))
            else:
                f.write(struct.pack('={}{}'.format(col, dtype), *mat))


def transfer_to_fvecs(pca_dim=128):
    vectors = np.load('imagenet_resnet_v1_50_vectors_pca{}.npy'.format(pca_dim))
    labels = np.load('imagenet_resnet_v1_50_lables.npy')
    save_fvec(vectors, 'imagenet_resnet_v1_50_vectors_pca{}.fvecs'.format(pca_dim), 'f')
    save_fvec(labels, 'imagenet.label', 'I')


def main():
    parser = argparse.ArgumentParser(description='Preprocess imagenet dataset for qvis')
    parser.add_argument('--datapath', type=str, help='location of imagenet dataset')
    parser.add_argument('--modelpath', type=str, help='location of tensorflow-slim model')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--pca', action='store_true')
    parser.add_argument('--fvec', action='store_true')
    args = parser.parse_args()

    if args.fvec:
        transfer_to_fvecs()
        return

    if args.pca:
        do_pca()
        return

    # infer resnet
    config = tf.ConfigProto()
    # config.operation_timeout_in_ms = 6000
    dataset = get_dataset(args.datapath)

    # from tensorflow.python.training import input as tf_input
    # from tensorflow.contrib.slim.python.slim.data import parallel_reader
    # data_files = parallel_reader.get_data_files(args.datapath)
    # print(len(data_files), 'files.')
    # filename_queue = tf_input.string_input_producer(data_files, num_epochs=1, shuffle=False, name='filenames')
    # reader = tf.TFRecordReader()
    # key, value = reader.read(filename_queue)
    # dvalue = dataset.decoder.decode(value)

    # with tf.Session(config=config) as sess:
    #     ini_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(ini_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)

    #     counter = 0
    #     while True:
    #         k, v = sess.run([key, dvalue])
    #         counter += 1
    #         # print(k, v)
    #         print(k, counter)

    # return

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=False, num_epochs=1)
    images, labels = provider.get(['image', 'label'])

    # import urllib
    # url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    # image_string = urllib.request.urlopen(url).read()
    # image = tf.image.decode_jpeg(image_string, channels=3)

    processed_images = preprocess_for_eval(images, 224, 224)
    # processed_images = tf.expand_dims(processed_images, 0)

    # Batch up
    processed_images, labels = tf.train.batch(
        [processed_images, labels],
        batch_size=args.batch_size,
        num_threads=8,
        capacity=2 * args.batch_size,
        allow_smaller_final_batch=True
    )

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, endpoints = resnet_v1.resnet_v1_50(processed_images, num_classes=1000, scope='resnet_v1_50', is_training=False)
        pool5 = math_ops.reduce_mean(endpoints['resnet_v1_50/block4'], [1, 2], name='pool5', keep_dims=True)
        vectors = tf.squeeze(pool5, axis=[1, 2])

    init_fn = slim.assign_from_checkpoint_fn(args.modelpath, slim.get_model_variables())

    vectors_to_save = []
    labels_to_save = []
    with tf.Session(config=config) as sess:
        ini_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(ini_op)
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(sess=sess, coord=coord)
        init_fn(sess)
        # prob = tf.squeeze(logits, axis=[1, 2])
        # probabilities = tf.nn.softmax(prob, dim=-1)
        counter = 0
        while True:
            try:
                vector, label = sess.run([vectors, labels])
            except OutOfRangeError as e:
                break
            print(vector.shape)
            vectors_to_save.append(vector)
            labels_to_save.append(label)
            counter += vector.shape[0]
            print(counter)
            # results, gtlabel = sess.run([probabilities, labels])
            # print(sorted(enumerate(results[0]), key=lambda x: -x[1])[:5], gtlabel)
        np.save("imagenet_resnet_v1_50_vectors.npy", np.concatenate(vectors_to_save))
        np.save("imagenet_resnet_v1_50_lables.npy", np.concatenate(labels_to_save))


if __name__ == '__main__':
    main()
