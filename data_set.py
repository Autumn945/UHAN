import numpy as np, sys, math, os, time
from pandas import DataFrame, Series
import pandas as pd, json, functools
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

data_home = 'data/'
col_names = ['pid',
             'uid',
             'postdate',
             'commentcount',
             'haspeople',
             'titlelen',
             'deslen',
             'tagcount',
             'avgview',
             'groupcount',
             'avgmembercount']

#wraps
def log(func):
    functools.wraps(func)
    def __wrap(*args, **kw):
        print('call %s()...' % func.__name__, end = '', flush = True)
        bt = time.time()
        ret = func(*args, **kw)
        ct = time.time() - bt
        print('\rcall %s() over, time: %.2f' % (func.__name__, ct))
        return ret
    return __wrap

def run_for(n = None, step = 0):
    def __d(func):
        def __w(d):
            if step == 0:
                return np.array([func(_d) for _d in d])
            bt = time.time()
            _n = len(d) if n is None else n
            def _f(i, _d):
                if i % step == 0:
                    nt = time.time() - bt
                    rt = (_n - i) * nt / i if i else 0
                    print('\r%s #%d/%d, cost %.2fs, rest: %.2fs(%.1fmin),  ' % (func.__name__, i, _n, nt, rt, rt / 60), end = '', flush = True)
                return func(_d)
            ret = np.array([_f(i, _d) for i, _d in enumerate(d)])
            print('over ~')
            return ret
        return __w
    return __d

def get_raw_data():
    json_fn = data_home + 'data.json'
    label_fn = data_home + 'train_label.txt'
    url_fn = data_home + 'train_url.txt'
    metadata_fn = data_home + 'train_data.txt'
    zone_fn = data_home + 'TimeZone.txt'
    cls = 'raw_pid raw_uid postdate commentcount haspeople titlelen deslen tagcount avgview groupcount avgmembercount'.split()

    data = pd.DataFrame([json.loads(s) for s in open(json_fn, 'r')])

    #del data['title']
    del data['img_url']
    del data['img_name']

    url = data.url.str.split('/')
    data['user'] = url.str.get(-2)
    data['post'] = url.str.get(-1)

    #help(pd.read_csv)
    metadata = pd.read_csv(metadata_fn, sep = ' ', header = None, names = cls)
    metadata['label'] = [float(l) for l in open(label_fn, 'r')]
    metadata.index = [u[:-1] for u in open(url_fn, 'r')]
    data = data.join(metadata, on = 'url')
    data['img'] = data_home + 'img/train/' + data.user + '/' + data.post + '.jpg'
    #print(data[:3]);return

    err_img = pd.read_csv(data_home + 'err_img.txt', names = ['img'])
    data = data[~data.img.isin(err_img.img)]
    data.desc = data.desc.str.strip()
    #print('raw:', len(data)); print(len(np.unique(data['user'].values)))
    data.desc = data.title.str.cat(data.desc, sep = ' #desc: ')
    #data = data[data.desc.str.len() > 0]


    zone = np.zeros((len(data), ), dtype = 'int32')
    g = data.groupby(['raw_uid'])
    for l in open(zone_fn, 'r'):
        l = l.strip()
        if l:
            u, z = l.split()
            z = int(z[:3])
            zone[g.indices[u]] = z
    data['zone'] = zone


    minute, week, day, seg = [], [], [], []
    year, month, a_m = [], [], []
    date = []
    h_map = [0] * 8 + [1] * 4 + [2] * 2 + [3] * 3 + [4] * 3 + [5] * 4
    for i, d in data.iterrows():
        t = time.gmtime(d.postdate + d.zone * 60 * 60)
        minute.append(t.tm_min)
        mday = t.tm_mday - 1
        week.append(mday // 7 + (mday % 7 > t.tm_wday))
        day.append(t.tm_wday)
        seg.append(h_map[t.tm_hour])
        date.append(time.strftime('%Y-%m-%d %H:%M:%S', t))

        year.append(t.tm_year - 2000)
        month.append(t.tm_mon - 1)
        a_m.append((t.tm_year - 2000) * 12 + t.tm_mon - 1)



    data['Tm'] = minute
    data['Td'] = day
    data['Tw'] = week
    data['Tp'] = seg
    data['date'] = date

    data['Tyear'] = year
    data['Tmonth'] = month
    data['Tam'] = a_m

    del data['url']
    return data

def get_data_from_cache(n = 0):
    cache_fn = data_home + 'cache_data.csv'
    if os.path.isfile(cache_fn) and not _update:
        print('using cached data, fn: %s' % cache_fn)
        data = pd.read_csv(cache_fn)
        if n: data = data[:n]
        return data
    data = get_raw_data()
    data = filter_data(data)
    data.to_csv(cache_fn, index = False)
    if n: data = data[:n]
    return data

def random_split_data(seed = 170705):
    global data, train, vali, test, nb_train, nb_vali, nb_test
    nb_vali = nb_items // 10
    nb_test = nb_items // 5
    nb_train = nb_items - nb_vali - nb_test
    _data = data.sample(frac = 1, random_state = seed)
    train = _data[:nb_train]
    vali = _data[nb_train: -nb_test]
    test = _data[-nb_test:]

def time_split_data(seed = 170705):
    global data, train, vali, test, nb_train, nb_vali, nb_test
    nb_vali = nb_items // 10
    nb_test = nb_items // 5
    nb_train = nb_items - nb_vali - nb_test
    data = data.sort_values('postdate')
    train = data[:nb_train]
    test = data[-(nb_test + nb_vali):]
    _test = test.sample(frac = 1, random_state = seed)
    test = _test[:nb_test]
    vali = _test[-nb_vali:]

def filter_data(data):
    vocab = CountVectorizer(min_df = 5)
    vocab.fit(data.desc.values)
    analyzer = vocab.build_analyzer()

    f = list(map(lambda s: all(ord(c) < 128 for c in s) and len(analyzer(s)) >= 5, data.desc))
    print(len(data))
    return data[f]

def get_data(n = 0):
    #print('getting data...')
    global data, test, nb_users, nb_items
    #data = get_raw_data()
    data = get_data_from_cache(n)
    #print(len(data))
    us = np.unique(data['user'].values)
    nb_users = len(us)
    nb_items = len(data)
    uid = dict(zip(us, range(nb_users)))
    data['uid'] = data['user'].map(uid)
    #del data['user']
    #del data['post']
    time_split_data()

def make_vocab():
    global count_vocab, text_analyzer, word_dict, nb_words
    from sklearn.feature_extraction.text import CountVectorizer
    count_vocab = CountVectorizer()
    count_vocab.fit(train.desc.values)
    text_analyzer = count_vocab.build_analyzer()
    word_dict = count_vocab.vocabulary_
    nb_words = len(word_dict)

def make_tfidf():
    global tfidf_vocab
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vocab = TfidfVectorizer(vocabulary = word_dict)
    tfidf_vocab.fit(train.desc.values)

def init(n = 0, update = 0, img_size = (448, 448)):
    global _update, _img_size
    _update = update
    _img_size = img_size
    get_data(n)
    make_vocab()
    make_tfidf()
    print('#u: %d, #p: %d, #wd: %d, #tr: %d, #va: %d, #ts: %d' %
            (nb_users, nb_items, nb_words, nb_train, nb_vali, nb_test))

@run_for(step = 100)
def load_vgg16(img):
    return np.load(img + '.vgg16.npy')

@run_for(step = 0)
def load_text_seq(s):
    return [word_dict[w] + 1 for w in text_analyzer(s) if w in word_dict]

@run_for(step = 0)
def load_img(img):
    from keras.preprocessing import image
    _img = image.load_img(img, target_size = _img_size)
    return image.img_to_array(_img)

@run_for(step = 0)
def load_vgg(img):
    return np.load(img + '.vgg16_196_512.npy')

def preprocessing_img():
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    from img_model import VGG16
    import util
    util.set_gpu_using(0.9, '2')
    model = VGG16()
    steps = 3
    _bt = time.time()
    imgs = data.img.values
    for i in range(nb_items):
        if i % steps == 0:
            _nt = time.time() - _bt
            _at = nb_items * _nt / i if i else 0
            print('\r#%d/%d, %.2fs/%.2fs, %.2fs' % (i, nb_items, _nt, _at, _at - _nt), end = '')
        fn = imgs[i] + '.vgg16_196_512.npy'
        if os.path.isfile(fn):
            continue
        r = load_img([imgs[i]])
        y = model.predict(r)
        np.save(fn, y.reshape((196, 512)))

def text_bow(text):
    return count_vocab.transform([' '.join(text_analyzer(s)[:50]) for s in text])

def text_tfidf(text):
    return tfidf_vocab.transform([' '.join(text_analyzer(s)[:50]) for s in text])

@log
def load_glove():
    global glove
    glove = {}
    for l in open(data_home + 'text/glove.6B.300d.txt', 'r'):
        w, *v = l.split()
        v = np.array(v, dtype = 'float32')
        glove[w] = v
#load_glove()

@run_for(step = 100)
def text_glove(text):
    t = text_analyzer(text)[:50]
    v = [glove[w] for w in t if w in glove]
    if not v:
        return np.zeros(50)
    return np.mean(v, 0)

def main():
    print('hello world, data_set')

if __name__ == '__main__':
    main()

