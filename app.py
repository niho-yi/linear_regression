import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import pandas as pd
from flask import Flask, render_template
# TF의 matplotlib의 한글 처리
from matplotlib import font_manager, rc
rc('font', family=font_manager.FontProperties(fname='C:/Windows/Fonts/H2GTRM.ttf').get_name())

from gradient_descent import *
from linear_regression import LinearRegression
#from k_mean import K_Mean
from k_mean_t import K_MeanT
from naive_bayse import NaiveBayseClassfier

app = Flask(__name__)

@app.route('/')
def index():
    #l = LinearRegression()
    #g = GradientDescent()
    #k = Kmean()
    nb = NaiveBayseClassfier()
    nb.train('data/review_train.csv')

    print("댓글이 긍정인지 부정인지 파악하라")
    check = nb.classify("너무 좋아요. 내 인생 최고의 명작 영화")
    print(check)
    #name = g.execute()
    return render_template('index.html', name=check)





if __name__ == '__main__':
    app.run()
