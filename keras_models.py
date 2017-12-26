import numpy as np, sys, math, os
import copy
import time
import keras
from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import concatenate
from keras.layers import add
from keras.layers import multiply

from keras.models import Model
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences

import tensorflow as tf

from models import MethodModel
import data_set
import logger
from logger import log, _timer
import AttLayers
import img_model
from img_model import ResNet50, VGG16, preprocess_input
import util

class Basic(MethodModel):
    def __init__(
            self,
            batch_size,
            epochs,
            max_brk,
            batch_steps,
            dim_k,
            maxlen,
            rnn,
            **kw):
        self.maxlen = maxlen
        self.batch_size = batch_size

        self.epochs = epochs
        self.max_brk = max_brk

        self.batch_steps = batch_steps
        self.dim_k = dim_k
        self.rnn = rnn

        super(Basic, self).__init__(**kw)
        log('all train data need %d epoches' % math.ceil(self.nb_train / (self.batch_size * self.batch_steps)))

    def init_model_args(
            self,
            fc = [],
            all_l2 = None,
            dropout = None,
            #clipnorm = None,
            **kw):
        super(Basic, self).init_model_args(**kw)
        self.fc = fc
        self.all_l2 = all_l2
        self.dropout = dropout

    def get_l2(self, name):
        if self.all_l2 is not None:
            return regularizers.l2(all_l2)
        return regularizers.l2(self.l2[name])

    def make_data(self, data):
        return util.keras_Data(
                data,
                maxlen = self.maxlen,
                batch_size = self.batch_size
                )

    def _fit(self):
        best_w, best_mse, brk = None, None, 0
        best_mae = None
        data_generator = self.train.generator(shuffle = True)
        for i in range(self.epochs):
            _timer.start()
            for _ in range(self.batch_steps):
                self.model.train_on_batch(*next(data_generator))
            mse = self.evaluate('vali', mtc = 'MSE')
            mae = self.evaluate('vali', mtc = 'MAE')
            #loss = self.evaluate('train')
            loss = -1
            if best_mse is None or mse < best_mse:
                best_mse = mse
                best_mae = mae
                best_w = self.model.get_weights()
                brk = 0
            else:
                brk += 1
            msg = '#%d/%d, vali mse: %.4f, mae: %.4f, brk: %d, time: %.2fs' % (i + 1, self.epochs, mse, mae, brk, _timer.stop())
            log(msg)
            if self.max_brk > 0 and brk >= self.max_brk:
                break
        _timer.start()
        self.model.set_weights(best_w)
        test_mse = self.evaluate('test', mtc = 'MSE')
        test_mae = self.evaluate('test', mtc = 'MAE')
        _timer.stop()
        return best_mse, test_mse, best_mae, test_mae

    def predict(self, x):
        return self.model.predict_generator(x.generator(), x.steps)

    def single_embedding(self, x, n, l2_name, dim_k = None):
        if dim_k is None:
            dim_k = self.dim_k
        x = Embedding(
                n,
                dim_k,
                embeddings_regularizer = self.get_l2(l2_name),
                )(x)
        x = Flatten()(x)
        if self.dropout is not None:
            x = Dropout(self.dropout)(x)
        return x

    def user_emb(self, dim_k = None):
        return self.single_embedding(
                self.inp_u, 
                data_set.nb_users,
                'user',
                dim_k)

    def text_embs(self):
        texts = Embedding(
                data_set.nb_words + 1,
                self.dim_k,
                embeddings_regularizer = self.get_l2('text'),
                mask_zero = True,
                )(self.inp_t)
        if self.dropout is not None:
            texts = Dropout(self.dropout)(texts)
        if self.rnn == 'LSTM':
            texts = LSTM(self.dim_k, return_sequences = True)(texts)
        if self.rnn == 'BiLSTM':
            texts = Bidirectional(LSTM(self.dim_k // 2, return_sequences = True))(texts)
        return texts

    def img_embs(self):
        i = TimeDistributed(
                Dense(
                    self.dim_k,
                    activation = 'tanh',
                    kernel_regularizer = self.get_l2('img'),
                    use_bias = False,
                    )
                )(self.inp_i)
        if self.dropout is not None:
            i = Dropout(self.dropout)(i)
        return i

    def post_emb(self):
        pass

    def make_inputs(self):
        self.inputs = []

        self.inp_u = Input(shape = (1,), name = 'user')
        self.inputs.append(self.inp_u)

        self.inp_t = Input(shape = (self.maxlen,), name = 'text')
        self.inputs.append(self.inp_t)

        self.inp_i = Input(shape = (14 * 14, 512), name = 'vgg')
        self.inputs.append(self.inp_i)

    def top_layers(self, x):
        for l in self.fc:
            x = Dense(
                    l,
                    activation = 'relu',
                    kernel_regularizer = self.get_l2('dense'),
                    bias_regularizer = self.get_l2('dense_b')
                    )(x)
        return x

    def make_model(self):
        self.make_inputs()
        x = self.post_emb()
        x = self.top_layers(x)
        y = Dense(
                1,
                kernel_regularizer = self.get_l2('dense'),
                bias_regularizer = self.get_l2('dense_b')
                )(x)
        model = Model(inputs = self.inputs, outputs = y)
        opt = optimizers.Adam()
        model.compile(
                optimizer = opt,
                loss = 'mse')
        return model

class UHAN(Basic):
    def post_emb(self):
        u = self.user_emb()
        t = self.text_embs()
        v = self.img_embs()
        t0 = AttLayers.Avg()(t)
        v0 = AttLayers.Avg()(v)
        t1 = AttLayers.Att6(l2 = self.get_l2('att'), name = 'att_text')([t, v0, u])
        v1 = AttLayers.Att6(l2 = self.get_l2('att'), name = 'att_image')([v, t0, u])
        o = AttLayers.MergeA(l2 = self.get_l2('merge'), name = 'merge')([t1, v1, u])
        Wu = Dense(self.dim_k, use_bias = False)(u)
        return add([o, Wu])

def main():
    print('hello world')

if __name__ == '__main__':
    main()
