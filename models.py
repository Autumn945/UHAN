import numpy as np, random
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from logger import log
from pprint import pprint, pformat

class Data:
    def __init__(self, data, get_x):
        self.r = data
        if 'label' in data:
            self.x = get_x(data.drop('label', axis = 1))
            self.y = data['label'].values
        else:
            self.x = get_x(data)

    def generator_data(self):
        return zip(self.x, self.y)

class MethodModel:
    def __init__(
            self,
            train,
            vali,
            test,
            run_times,
            name = None,
            **kw
            ):
        self.run_times = run_times
        self.raw_data = (train, vali, test)
        self.name = type(self).__name__ if name is None else name

        self.nb_train, self.nb_vali, self.nb_test = list(map(len, [train, vali, test]))
        self.train = self.make_data(train)
        self.vali = self.make_data(vali)
        self.test = self.make_data(test)

        self.model = None

    def make_data(self, data):
        return Data(data, self.get_x)

    def fit(self, args):
        vali_mse, test_mse = [], []
        vali_mae, test_mae = [], []
        for _ in range(self.run_times):
            log('run_times#%d/%d' % (_ + 1, self.run_times))
            np.random.seed(123456)
            self.init_model_args(**args)
            self.model = self.make_model()
            _v_s, _t_s, _v_a, _t_a = self._fit()
            vali_mse.append(_v_s)
            test_mse.append(_t_s)
            vali_mae.append(_v_a)
            test_mae.append(_t_a)
            msg = 'v_s: %.4f, v_a: %.4f, t_s: %.4f, t_a: %.4f' % (_v_s, _v_a, _t_s, _t_a)
            log(msg, red = True)
        return np.mean(vali_mse), np.mean(vali_mae), np.mean(test_mse), np.mean(test_mae)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y = None, mtc = 'xxx'):
        if y is None:
            if type(x) == str:
                x = eval('self.' + x)
            x, y = x.x, x.y
        return self.metric(self.predict(x), y, mtc)

    def save_model(self, fn = 'temp'):
        pass

    def init_model_args(self, **kw):
        self.model_args = {}
        self.l2 = {}
        for key, value in kw.items():
            if key[-3:] == '_l2':
                self.l2[key[:-3]] = value
            else:
                self.model_args[key] = value

    def make_model(self, **kw):
        pass

    @staticmethod
    def load_model(fn = 'temp'):
        return None

    @staticmethod
    def metric(y1, y2, mtc):
        y1, y2 = y1.reshape((-1,)), y2.reshape((-1,))
        assert y1.shape == y2.shape, 'same shape, found %s and %s.' % (str(y1.shape), str(y2.shape))
        return eval(mtc)(y1, y2)


