import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import Flask, render_template
# TF의 matplotlib의 한글 처리
from matplotlib import font_manager, rc
rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/H2GTRM.ttf').get_name())




def gradient_descent():
    X = [1., 2., 3.]
    Y = [1., 2., 3.]
    m = n_sample = len(X)
    W = tf.placeholder(tf.float32)
    hypothesis = tf.multiply(X, W)
    cost = tf.reduce_mean(tf.pow(hypothesis-Y, 2)) / m
    W_val = []
    cost_val = []
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(-30, 50):
        W_val.append(i * 0.1)
        cost_val.append(sess.run(cost, feed_dict={W: i*0.1}))

    plt.plot(W_val, cost_val, 'ro')
    plt.ylabel('COST-1')
    plt.xlabel('W-1')
    return 'gradient_decent.svg'
