import numpy as np, sys, math, os, random, json, time
import argparse, atexit
np.random.seed(170831)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import models
import data_set, logger, util
import importlib
from collections import ChainMap
from pprint import pprint, pformat
import keras_models
from logger import log

def run(**kwargs):
    log('start')
    Model = keras_models.UHAN
    model = Model(
            train = data_set.train,
            vali = data_set.vali,
            test = data_set.test,
            **kwargs)
    args = tc()
    r = model.fit(args)
    print('vali: mse={:.3f}, mae={:.3f}, test: mse={:.3f}, mae={:.3f}'.format(*r))

def tc():
    a = {}
    a['dense_l2'] = 0
    a['dense_b_l2'] = 1e-4
    a['user_l2'] = 0
    a['text_l2'] = 1e-4
    a['img_l2'] = 0
    a['att_l2'] = 0
    a['merge_l2'] = 0
    a['fc'] = [1024, 1024]
    return a

def _main(gpu, **model_args):

    data_set.init(update = 0, img_size = (448, 448))

    util.set_gpu_using(0.9, gpu)
    run(**model_args)


def main():
    print('hello world, main')
    parser = argparse.ArgumentParser(description = 'models')

    default_args = {}
    short = {}
    def add_argument(*v, **kw):
        parser.add_argument(*v, **kw)
        key = v[-1].replace('-', '')
        short[key] = v[0].replace('-', '')
        if 'default' in kw:
            default_args[key] = kw['default']

    add_argument('-gpu', type = str, default = '3')

    add_argument('-rt', '--run_times', type = int, default = 3)
    add_argument('-ep', '--epochs', type = int, default = 5000)
    add_argument('-k', '--dim_k', type = int, default = 512)
    add_argument('-rnn', type = str, default = 'LSTM')
    add_argument('-maxlen', type = int, default = 50)
    add_argument('-bs', '--batch_steps', type = int, default = 64)
    add_argument('-batch_size', type = int, default = 128)
    add_argument('-max_brk', type = int, default = 20)


    args = parser.parse_args()
    #print(pformat(args.__dict__)); return
    for key, value in args.__dict__.items():
        if key in default_args and value != default_args[key]:
            log_fn.append(short[key] + ':' + str(value))
    logger.log(args)
    _main(**args.__dict__)


if __name__ == '__main__':
    main()

