import numpy as np, sys, math, os

os.environ['CUDA_VISIBLE_DEVICE'] = '0,1,2,3'
def set_gpu_using(memory_rate = 0.9, gpus = '0'):  
    """ 
    This function is to allocate GPU memory a specific fraction 
    """  
    from keras import backend as K
    import tensorflow as tf
    K.clear_session()
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = memory_rate,
            visible_device_list = gpus,
            allow_growth = True,
            )
  
    if num_threads:
        session = tf.Session(
                config = tf.ConfigProto(
                    gpu_options = gpu_options,
                    intra_op_parallelism_threads = num_threads))
    else:
        session = tf.Session(
                config = tf.ConfigProto(
                    gpu_options = gpu_options))
    K.set_session(session)

class keras_Data:
    global data_set
    import data_set
    def __init__(self, data, batch_size, maxlen = None):
        self.data = data
        self.len = len(data)
        self.steps = (self.len + batch_size - 1) // batch_size

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.x = self
        self.y = data.label.values

    @staticmethod
    def one_hot(v, n):
        matrix = np.eye(n)
        return matrix[v]

    @staticmethod
    def user(batch):
        u = batch.uid.values
        u = u.reshape((-1, 1))
        return u
    @staticmethod
    def text(batch):
        t = batch.desc.values
        from keras.preprocessing.sequence import pad_sequences
        t = data_set.load_text_seq(t)
        t = pad_sequences(t, maxlen = 50, padding = 'pre', truncating = 'post')
        return t
    @staticmethod
    def image(batch):
        i = batch.img.values
        i = data_set.load_vgg(i)
        return i
    
    @staticmethod
    def get_batch(batch):
        x = []
        x.append(keras_Data.user(batch))
        x.append(keras_Data.text(batch))
        x.append(keras_Data.image(batch))
        return x, batch.label.values

    def get_data(self):
        return self.get_batch(self.data)

    def generator(self, shuffle = False):
        while 1:
            idx = list(range(self.len))
            if shuffle:
                np.random.shuffle(idx)
            for i in range(0, self.len, self.batch_size):
                batch = self.data.iloc[idx[i: i + self.batch_size]]
                yield self.get_batch(batch)

def main():
    print('hello world, util.py')

if __name__ == '__main__':
    main()

