import tensorflow as tf  
from tensorflow.keras import layers  

class Evaluation:

    def __init__(self, strategy=None):  
        self.strategy = strategy
        H, W, C = 10, 10, 3
        imgs = tf.zeros([8, H, W, C])
        self.dataset = tf.data.Dataset.from_tensor_slices(imgs)
        self.dataset = self.dataset.batch(4)
        self.dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        with self.strategy.scope():
            self.conv1 = layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')
            self.conv2 = layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same')

    @tf.function
    def serving(self, img):
        prediction1 = self.conv1(img)
        prediction2 = self.conv2(img)
        return {
            'pre1': prediction1,
            'pre2': prediction2,
        }

    @tf.function
    def infer(self, serve_summary_writer, key_list):
        with serve_summary_writer.as_default():
            batch = tf.cast(0, tf.int64)
            for img in self.dataset:
                prediction_perReplica = strategy.run(self.serving, args=(img,))
                tf.print("prediction_perReplica:", prediction_perReplica)
                for key in key_list:
                    prediction_tensor = prediction_perReplica[key].values
                    prediction_concat = tf.concat(prediction_tensor, axis = 0)
                    tf.summary.write(tag="prediction_" + key, tensor=prediction_concat, step=batch)
                batch += 1
                
    def eval(self):
        serve_summary_writer = tf.summary.create_file_writer('tmp', max_queue=100000, flush_millis=100000)
        key_list = ["pre1", "pre2"]
        self.infer(serve_summary_writer, key_list)
        serve_summary_writer.close()
        tf.io.gfile.rmtree('tmp')  

if __name__ == "__main__":

    strategy = tf.distribute.MirroredStrategy()
    e = Evaluation(strategy)   
    e.eval()

