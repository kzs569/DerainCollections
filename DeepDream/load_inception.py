# coding:utf-8
from __future__ import print_function
import numpy as np
import tensorflow as tf

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

model_fn = 'tensorflow_inception_graph.pb'
with tf.gfile.FastGFile(model_fn,'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0


