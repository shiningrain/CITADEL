2021-02-20 01:42:11.421182: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
2021-02-20 01:42:11.424571: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2194875000 Hz
WARNING:tensorflow:/job:worker/replica:0/task:0 seems down, retrying 1/3
WARNING:tensorflow:/job:worker/replica:0/task:0 seems down, retrying 2/3
ERROR:tensorflow:Cluster check alive failed, /job:worker/replica:0/task:0 is down, aborting collectives: Deadline Exceeded
Additional GRPC error information from remote target /job:worker/replica:0/task:0:
:{"created":"@1613785391.258566931","description":"Deadline Exceeded","file":"external/com_github_grpc_grpc/src/core/ext/filters/deadline/deadline_filter.cc","file_line":69,"grpc_status":4}


And here is my code to reproduce this issue:
python
def test_multi_worker_mirrored_strategy_send_recv(args):
    # specify cluster resolver
    import os, json

    # set visible gpus
    os.environ['CUDA_VISIBLE_DEVICES']= ""

    # set log leverl
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 show all, 2 for warning and error, 3 for only error

    port = "12345"
    ips = [host1, host2]
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': { "worker": [ip + ":" + port for ip in ips] },
        "task" : {'type': 'worker', "index": args.task_id}
    })
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()

    # create stratey
    options = tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.AUTO)
    strategy = tf.distribute.MultiWorkerMirroredStrategy(resolver, options)

    print("num_accelerators = ", strategy.cluster_resolver.num_accelerators())
    print("task_type = %s, task_id = %s" %(strategy.cluster_resolver.task_type, strategy.cluster_resolver.task_id))

    @tf.function
    def _test_step(task_id):
        send_device = "/job:worker/replica:0/task:0/device:CPU:0"
        recv_device = "/job:worker/replica:0/task:1/device:CPU:0"
        if task_id == 0: # send
            tensor = tf.constant([1.0, 2.0], dtype=tf.float32)
            tf.raw_ops.Send(tensor=tensor, 
                            tensor_name="test_send_tensor", 
                            send_device=send_device, 
                            send_device_incarnation=123, 
                            recv_device=recv_device,
                            client_terminated=True)
            print("Send tensor from: ", tensor.device)
        else: # recv
            tensor = tf.raw_ops.Recv(tensor_type=tf.float32, 
                                     tensor_name="test_send_tensor", 
                                     send_device=send_device, 
                                     send_device_incarnation=123, 
                                     recv_device=recv_device,
                                     client_terminated=False)
            print("Recv tensor from: ", tensor.device)
        return tensor

    tensor = strategy.run(_test_step, args=(args.task_id,))
    print(tensor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--task_id', type=int, 
                        help='specify the task id in Multi Worker Mirrored Strategy cluster.',
                        required=True)
    args = parser.parse_args()

    test_multi_worker_mirrored_strategy_send_recv(args)

