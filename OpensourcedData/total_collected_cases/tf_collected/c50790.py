
import tensorflow as tf
from mpi_cluster_resolver import MPIClusterResolver

resolver = MPIClusterResolver()
strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=resolver)

