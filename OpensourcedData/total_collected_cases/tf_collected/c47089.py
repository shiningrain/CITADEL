import tensorflow as tf


class MyMetric(tf.keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = self.add_weight(name='w', shape=(1,))

    def update_state(self, w):
        self.w.assign(w)

    def result(self):
        return self.w

m = MyMetric()
m.reset_states()

