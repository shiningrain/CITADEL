import time
from multiprocessing import Process

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.training.server_lib import Server

CLUSTER_SPEC = {
    "worker": [
        "localhost:14286",
        "localhost:14287"
    ]
}

GROUP_SIZE = 4


def _configure(group_size):
    gpu_options = config_pb2.GPUOptions(
        visible_device_list='0,1',
        per_process_gpu_memory_fraction=0.7 / group_size
    )
    experimental = config_pb2.ConfigProto.Experimental(collective_nccl=True)
    experimental.collective_group_leader = '/job:worker/replica:0/task:0'
    return config_pb2.ConfigProto(gpu_options=gpu_options, experimental=experimental)


class TFCluster:
    def __init__(self, cluster_spec):
        self._cluster_spec = cluster_spec
        self._num_worker = len(self._cluster_spec.get("worker", []))
        self._tf_servers = []

    def start(self):
        def server(job_name: str, task_index: int):
            s = Server(self._cluster_spec,
                       job_name=job_name,
                       task_index=task_index,
                       config=_configure(GROUP_SIZE))
            s.join()

        assert self._num_worker >= 1
        for i in range(self._num_worker):
            self._tf_servers.append(Process(target=server,
                                            args=("worker", i), daemon=True))
            # break
        for proc in self._tf_servers:
            proc.start()

    def stop(self):
        for proc in self._tf_servers:
            proc.terminate()


if __name__ == '__main__':
    cluster = TFCluster(CLUSTER_SPEC)
    cluster.start()
    time.sleep(5)
    input('Press Enter to Stop.')
    cluster.stop()

`task.py`:
python
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import ops
from tensorflow.python.client.session import Session
from tensorflow.python.ops import collective_ops

from cluster import CLUSTER_SPEC, GROUP_SIZE

VAR = np.array([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1])
VAR_TASK_INDEX = 0


def test_collective(job_name, task_index, num_gpus):
    worker_device = "/job:%s/task:%d" % (job_name, task_index)
    master_target = "grpc://" + CLUSTER_SPEC[job_name][task_index]
    print('> Session Target:', master_target)

    with ops.Graph().as_default(), Session(target=master_target) as sess:
        def run(x):
            run_options = config_pb2.RunOptions()
            run_options.experimental.collective_graph_key = task_index + 1
            # Different positive graph key for different task to avoid racing conditions.
            return sess.run(x, options=run_options)

        with ops.device('/job:worker/task:%d/device:CPU:0' % VAR_TASK_INDEX):  # make sure all use the same variable
            var = tf.Variable(VAR, name='W')

        targets = []
        collectives = []
        for i in range(num_gpus):
            with ops.device(worker_device + '/device:GPU:' + str(i)):
                t = var + 0.2 * task_index + 0.1 * i
                targets.append(t)
                collectives.append(
                    collective_ops.all_gather(
                        t,
                        group_size=GROUP_SIZE,
                        group_key=1, instance_key=1
                    )
                    # collective_ops.all_reduce(
                    #     t,
                    #     group_size=GROUP_SIZE,
                    #     group_key=1, instance_key=1, merge_op='Add', final_op='Div'
                    # )
                )

        run(tf.compat.v1.global_variables_initializer())

        var_value = run(var)
        print('> Variable Value:', var_value)

        targets_value = run(targets)
        print('> Targets Value:', targets_value)

        collectives_value = run(collectives)
        print('> Collectives Value:', collectives_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    parser.add_argument(
        "--job_name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
    )
    FLAGS, unparsed = parser.parse_known_args()
    num_gpus_per_node = 2
    test_collective(job_name=FLAGS.job_name, task_index=FLAGS.task_index, num_gpus=num_gpus_per_node)

