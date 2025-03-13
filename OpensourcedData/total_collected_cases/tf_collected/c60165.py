When either using tf.estimator.BestExporter or using tf.compat.v1.gfile.Rename directly when the source and destination are folders in S3, an error is thrown. This error seems to go back to Tensorflow 2.6 when the S3 support was moved into tensorflow_io. In Tensorflow 2.5 this behaves properly. It also functions properly if the folders are local.

Right now if you are using Tensorflow >= 2.6 and using a tf.estimator.BestExporter with the output being written to S3, an exception is thrown



### Standalone code to reproduce the issue

shell
import tensorflow as tf
import tensorflow_io as tfio

SOURCE_DIR = 's3://.../best_exporter/1'
DEST_DIR = 's3://.../best_exporter/old'

tf.compat.v1.gfile.Rename(SOURCE_DIR, DEST_DIR)

