TypeError: Fetch argument PerReplica:{
  0 /replica:0/task:0/device:GPU:0: <tf.Tensor 'Sub:0' shape=() dtype=float32>,
  1 /replica:0/task:0/device:GPU:1: <tf.Tensor 'replica_1/Sub:0' shape=() dtype=float32>
} has invalid type <class 'tensorflow.python.distribute.values.PerReplica'>, must be a string or Tensor. (Can not convert a PerReplica into a Tensor or Operation.)


Non-distributed evaluation (that is, with `RunConfig.eval_distribute=None` or with a single GPU only) finishes without errors.

**Standalone code to reproduce the issue** 
python
import numpy as np
import tensorflow as tf

def model_fn(features, labels, mode):
    predictions = tf.layers.dense(features, 2)
    metrics = {'cos': tf.metrics.mean_cosine_distance(labels, predictions, 1)}
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=tf.constant(0.1),
        train_op=None,
        eval_metric_ops=metrics)


def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        (np.array([[1., 1.]]), np.array([[2., 2.]])))
    dataset = dataset.repeat()
    dataset = dataset.batch(1, drop_remainder=True)
    return dataset


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 1, 'Need >1 GPUs to run'
    strategy = tf.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(train_distribute=strategy,
                                        eval_distribute=strategy)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    print(estimator.evaluate(input_fn, steps=5))

