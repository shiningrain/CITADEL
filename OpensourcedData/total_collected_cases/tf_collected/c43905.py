import os
import datetime
from tqdm import tqdm

import numpy as np

import tensorflow as tf
print('TF version', tf.__version__)


@tf.function
def do_stuff(wmat, tf_var):

    A = tf.matmul(wmat + 1, tf.transpose(wmat))
    error = tf.reduce_mean(tf_var)
    return error, A 

exp_uuid = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

n_batches = 20

weights = [None] * n_batches
for i in range(n_batches):
    weights[i] = tf.constant(np.random.rand(2000,10000), dtype=tf.float32)


def gen():
    for i in weights:
        yield i

option_names = ['prefetch_gpu', 'copy', 'copy_prefetch', 'copy_prefetch_gpu']
for option in range(4):

    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32))

    if option == 0:
        ## Option 1: prefetch_gpu
        #
        ## output:  
        ##          weights device /job:localhost/replica:0/task:0/device:CPU:0
        ##          weights device after identity /job:localhost/replica:0/task:0/device:GPU:0

        gpu_transform = tf.data.experimental.prefetch_to_device('/gpu:0')
        dataset.apply(gpu_transform)

    elif option == 1:
        ## Option 1: only copy
        #
        ## output:
        ##          weights device /job:localhost/replica:0/task:0/device:GPU:0
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))

    elif option == 2:
        ## Option 2: copy + prefetch
        ## as suggested in https://github.com/tensorflow/tensorflow/issues/35563#issuecomment-602160568
        #
        ## output:
        ##          weights device /job:localhost/replica:0/task:0/device:GPU:0

        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        with tf.device("/gpu:0"):
            dataset = dataset.prefetch(1)

    elif option == 3:
        ## Option 3: copy + prefetch_gpu
        #
        ## output:
        ##          weights device /job:localhost/replica:0/task:0/device:GPU:0
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        gpu_transform = tf.data.experimental.prefetch_to_device('/gpu:0')
        dataset.apply(gpu_transform)


    tf_var = tf.Variable(np.zeros(3))
    adam = tf.keras.optimizers.Adam(1e-4) 
    logpath = os.path.join('data', 'logs', 'pa_' + exp_uuid + '_' + option_names[option])

    tf.profiler.experimental.start(logpath)
    start = datetime.datetime.now()
    for b, wmat in tqdm(enumerate(dataset)):
        with tf.GradientTape() as tape:

            if b == 0:
                print('\n weights device', wmat.device)
                print('')

            if option == 0:
                wmat = tf.identity(wmat, 'move_to_gpu')
                if b == 0:
                    print('weights device after identity', wmat.device)
                    print('')

            # Do some calculations
            result = do_stuff(wmat, tf_var)
        
        grads = tape.gradient(result[0], [tf_var])
        adam.apply_gradients(zip(grads, [tf_var]))
    stop = datetime.datetime.now()
    tf.profiler.experimental.stop()

    print(f'\nOption {option_names[option]}')
    print(logpath)
    print('Time lapsed=', stop - start)

