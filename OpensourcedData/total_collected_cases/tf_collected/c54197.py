import tensorflow as tf
from datetime import datetime

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self):
        #N must be divisible by 4
        super(CustomLayer, self).__init__()
        self.dense1 = tf.keras.layers.Dense(100,use_bias=False)

    @tf.function
    def call(self, query):
        c = tf.cast(query,tf.int32,name="castQueryToInt32")
        d = tf.expand_dims( c, axis=1,name="expandDimsAxis1")
        e = tf.expand_dims( c, axis=2,name="expandDimsAxis2")
        g = tf.add(d , e ,name="additionBroadcasted")
        #h = tf.reduce_sum(g, axis=1) #This also create extra memory transfer between gpu and cpu
        f = tf.cast(g,tf.float32, name="castGToFloat")
        f = tf.reduce_sum(f,axis=1,name="reduceSumF")
        rem = (query - f)* (query - f)
        out = self.dense1(rem)
        return out



inputs = tf.keras.Input(shape=(100,))
out = CustomLayer()(inputs)


model = tf.keras.Model(inputs=inputs, outputs=out)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),loss="mse")


xtrain = tf.random.uniform((600000,100),dtype=tf.float32)
ytrain = tf.random.uniform((600000,100),dtype=tf.float32)

logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

#We investigate the slowness due the extra memory transfer between GPU and CPU because of broadcasting behavior of integer tensors
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 profile_batch="10,20")

model.fit( xtrain,ytrain, batch_size=100,epochs=1,callbacks=[tboard_callback])

