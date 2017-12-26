import numpy as np, sys, math, os
from keras.layers import Layer
from keras import backend as K
import tensorflow as tf
from logger import log

class Avg(Layer):
    def __init__(self, return_sequences = False, **kw):
        self.return_sequences = return_sequences
        super().__init__(**kw)

    def call(self, x, mask = None):
        # x: (None, maxlen, dim_k), mask: (None, maxlen)
        if mask is not None:
            mask = K.cast(mask, 'float32')
            x = x * K.expand_dims(mask, -1)
            x = K.sum(x, -2) / K.expand_dims(K.sum(mask, -1), -1)
            return x
        else:
            return K.mean(x, -2)

    def att_output(self, c, att, mask = None):
        att = K.softmax(att)
        if mask is not None:
            att = att * K.cast(mask, 'float32')
            att = att / K.expand_dims(K.sum(att, -1), -1)

        self.att_value = att
        att_text = c * K.expand_dims(att, -1)
        if self.return_sequences:
            return att_text
        return K.sum(att_text, -2)

    def compute_mask(self, inputs, mask = None):
        if type(mask) == list:
            mask = mask[0]
        if self.return_sequences:
            return mask
        return None

    def compute_output_shape(self, inputs):
        if type(inputs) == list:
            inputs = inputs[0]
        if self.return_sequences:
            return inputs
        return (inputs[0], inputs[-1])

BasicAtt = Avg

class Basic_add_W(BasicAtt):
    def __init__(self, l2 = None, **kw):
        self.l2 = l2
        super().__init__(**kw)

    def add_weight(self, name, shape, add_l2 = False):
        if add_l2:
            return super().add_weight(name, shape, regularizer = self.l2, initializer = 'glorot_normal')
        else:
            return super().add_weight(name, shape, initializer = 'glorot_normal')

# wh * (tanh(wt * t) * tanh(wa * a) * tanh(wb * b))
class Att6(Basic_add_W):
    def build(self, input_shape):
        maxlen = input_shape[0][1]
        dim_k = input_shape[0][2]
        dim_h = dim_k
        self.wt = self.add_weight('wt', (dim_k, dim_h), add_l2 = True)
        self.wa = self.add_weight('wa', (dim_k, dim_h), add_l2 = True)
        self.wb = self.add_weight('wb', (dim_k, dim_h), add_l2 = True)
        self.wh = self.add_weight('wh', (dim_h, 1), add_l2 = True)
        super().build(input_shape)

    def call(self, x, mask = None):
        # t: (None, maxlen, dim_k), u: (None, dim_k), mask: (None, maxlen)
        t, a, b = x
        maxlen = K.shape(t)[1]
        # (None, maxlen, dim_h)
        wt = K.tanh(K.dot(t, self.wt))
        # (None, dim_h)
        wa = K.tanh(K.dot(a, self.wa))
        wb = K.tanh(K.dot(b, self.wb))
        h = wt * K.expand_dims(wa * wb, 1)
        att = K.reshape(K.dot(h, self.wh), (-1, maxlen))
        #att = K.tanh(att)
        if type(mask) == list:
            mask = mask[0]
        return self.att_output(t, att, mask)


class BasicMerge(Layer):
    def __init__(self, l2 = None, **kw):
        self.l2 = l2
        super().__init__(**kw)

    def add_weight(self, name, shape, add_l2 = False):
        return super().add_weight(name, shape, regularizer = self.l2, initializer = 'glorot_normal')

    def compute_mask(self, inputs, mask = None):
        return None

    def compute_output_shape(self, inputs):
        inputs = inputs[0]
        return (inputs[0], inputs[-1])

class MergeA(BasicMerge):
    def build(self, input_shape):
        *x, u = input_shape
        n = len(x)
        #print(n); print(x[0])
        dim_k = x[0][-1]
        dim_h = dim_k
        #self.w = self.add_weight('w', (n, dim_k, dim_h), add_l2 = True)
        self.w = [self.add_weight('w%d' % i, (dim_k, dim_h), add_l2 = True) for i in range(n)]
        self.wu = self.add_weight('wu', (dim_k, dim_h), add_l2 = True)
        self.wh = self.add_weight('wh', (dim_h, 1), add_l2 = True)
        self.n = n
        self.dim_k = dim_k
        super(eval(type(self).__name__), self).build(input_shape)

    def call(self, x, mask = None):
        # (None, dim_k)
        *x, u = x
        wu = K.tanh(K.dot(u, self.wu))
        a = []
        for i in range(self.n):
            wx = K.tanh(K.dot(x[i], self.w[i]))
            h = wx * wu
            _a = K.dot(h, self.wh)
            _a = K.reshape(_a, (-1, 1))
            _a = K.exp(_a)
            a.append(_a)
        sa = sum(a)
        o = []
        self.att_value = []
        for i in range(self.n):
            _a = a[i] / sa
            self.att_value.append(_a)
            _o = _a * x[i]
            o.append(_o)
        return sum(o)

def get_att_layer(i):
    if i == 0:
        return Basic
    return eval('Att' + str(i))

def main():
    print('hello world, AttLayers.py')

if __name__ == '__main__':
    main()

