
def train(self, num_epoch = 10, gpu = True):
        
        if gpu : 
            CUDA_LAUNCH_BLOCKING="1"

            #torch.set_default_tensor_type(torch.FloatTensor) 
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda:0" if use_cuda else "cpu")
            model.to(device)
            if self.multi_object_detection == False : 
                num_classes = 2 # ['Tool', 'background']
            else : 
                print("need to set a multi object detection code")

            in_features = torch.tensor(model.roi_heads.box_predictor.cls_score.in_features, dtype = torch.int64).to(device)
            print("in_features = {}".format(in_features))
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            print( "model.roi_heads.box_predictor {}".format( model.roi_heads.box_predictor))
            
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            #params = sum([np.prod(p.size()) for p in model_parameters])
            params = [p for p in model.parameters() if p.requires_grad]

            
            optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            gc.collect()
            num_epochs = 5
            FILE_model_dict_gpu = "model_state_dict__gpu_lab2_and_lab7_5epoch.pth"
            list_of_list_losses = []
            print("device = ", device)
            
            if (self.data_loader.dataset) == None :
                self.build_dataloader(device)
            
            for epoch in tqdm(range(num_epochs)):

                # Train for one epoch, printing every 10 iterations
                train_his_, list_losses, list_losses_dict = train_one_epoch(model, optimizer, self.data_loader, device, epoch, print_freq=10)
                lr_scheduler.step()



### my train_one_epoch function (I print some information that will be shown on the output at the end of the message)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):

    model.train()
    metric_logger = utilss.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utilss.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    list_losses = []
    list_losses_dict = []
    for i, values in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header))):
        images, targets = values
        for image in images : 
            print("before the to(device) operation, image.is_cuda = {}".format(image.is_cuda))
        images = list(image.to(device, dtype=torch.float) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #images = [image.cuda() for image in images]
        for image in images : 
            print(image)
            print("after the to(device) operation, image.is_cuda = {}".format(image.is_cuda))
        for target in targets :
            for t, dict_value in target.items():
                print("after the to(device) operation, dict_value.is_cuda = {}".format(dict_value.is_cuda))

        print("images = {}".format(images))
        print("targets = {}".format(targets))

        # Feed the training samples to the model and compute the losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())



#### My output with the error


in_features = 1024
model.roi_heads.box_predictor FastRCNNPredictor(
  (cls_score): Linear(in_features=1024, out_features=2, bias=True)
  (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
)
device =  cuda:0

before the to(device) operation, image.is_cuda = True
tensor([[[0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         [0.0235, 0.0235, 0.0235,  ..., 0.0235, 0.0235, 0.0235],
         [0.0353, 0.0353, 0.0353,  ..., 0.0314, 0.0314, 0.0314]],

        [[0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.0078, 0.0078, 0.0078,  ..., 0.0039, 0.0039, 0.0039],
         [0.0235, 0.0235, 0.0235,  ..., 0.0157, 0.0157, 0.0157],
         [0.0353, 0.0353, 0.0353,  ..., 0.0235, 0.0235, 0.0235]],

        [[0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         [0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         [0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         ...,
         [0.0078, 0.0078, 0.0078,  ..., 0.0078, 0.0078, 0.0078],
         [0.0235, 0.0235, 0.0235,  ..., 0.0196, 0.0196, 0.0196],
         [0.0353, 0.0353, 0.0353,  ..., 0.0275, 0.0275, 0.0275]]],
       device='cuda:0')
after the to(device) operation, image.is_cuda = True
after the to(device) operation, dict_value.is_cuda = True
after the to(device) operation, dict_value.is_cuda = True
after the to(device) operation, dict_value.is_cuda = True
after the to(device) operation, dict_value.is_cuda = True
after the to(device) operation, dict_value.is_cuda = True
images = [tensor([[[0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         [0.0235, 0.0235, 0.0235,  ..., 0.0235, 0.0235, 0.0235],
         [0.0353, 0.0353, 0.0353,  ..., 0.0314, 0.0314, 0.0314]],

        [[0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         [0.0078, 0.0078, 0.0078,  ..., 0.0000, 0.0000, 0.0000],
         ...,
         [0.0078, 0.0078, 0.0078,  ..., 0.0039, 0.0039, 0.0039],
         [0.0235, 0.0235, 0.0235,  ..., 0.0157, 0.0157, 0.0157],
         [0.0353, 0.0353, 0.0353,  ..., 0.0235, 0.0235, 0.0235]],

        [[0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         [0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         [0.0078, 0.0078, 0.0078,  ..., 0.0118, 0.0118, 0.0118],
         ...,
         [0.0078, 0.0078, 0.0078,  ..., 0.0078, 0.0078, 0.0078],
         [0.0235, 0.0235, 0.0235,  ..., 0.0196, 0.0196, 0.0196],
         [0.0353, 0.0353, 0.0353,  ..., 0.0275, 0.0275, 0.0275]]],
       device='cuda:0')]
targets = [{'boxes': tensor([[1118.8964,    0.0000, 1368.9186,  399.3243],
        [1043.0958,  111.4863, 1332.4319,  426.1295]], device='cuda:0',
       dtype=torch.float64), 'labels': tensor([1, 1], device='cuda:0'), 'index': tensor([311], device='cuda:0'), 'area': tensor([99839.9404, 91037.6485], device='cuda:0', dtype=torch.float64), 'iscrowd': tensor([0], device='cuda:0')}]

/home/nathaneberrebi/anaconda3/lib/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448278899/work/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
0it [00:02, ?it/s]
  0%|          | 0/5 [00:02<?, ?it/s]

---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-15-51a35da5b1fe> in <module>
----> 1 class_model.train()

<ipython-input-7-d44d099a7743> in train(self, num_epoch, gpu)
    144 
    145                 # Train for one epoch, printing every 10 iterations
--> 146                 train_his_, list_losses, list_losses_dict = train_one_epoch(model, optimizer, self.data_loader, device, epoch, print_freq=10)
    147                 list_of_list_losses.append(list_losses)
    148                 # Compute losses over the validation set

<ipython-input-6-347c12a81a2f> in train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
    519 
    520         # Feed the training samples to the model and compute the losses
--> 521         loss_dict = model(images, targets)
    522         losses = sum(loss for loss in loss_dict.values())
    523 

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/torchvision/models/detection/generalized_rcnn.py in forward(self, images, targets)
     95             features = OrderedDict([('0', features)])
     96         proposals, proposal_losses = self.rpn(images, features, targets)
---> 97         detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
     98         detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
     99 

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/torchvision/models/detection/roi_heads.py in forward(self, features, proposals, image_shapes, targets)
    752         box_features = self.box_roi_pool(features, proposals, image_shapes)
    753         box_features = self.box_head(box_features)
--> 754         class_logits, box_regression = self.box_predictor(box_features)
    755 
    756         result: List[Dict[str, torch.Tensor]] = []

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/torchvision/models/detection/faster_rcnn.py in forward(self, x)
    280             assert list(x.shape[2:]) == [1, 1]
    281         x = x.flatten(start_dim=1)
--> 282         scores = self.cls_score(x)
    283         bbox_deltas = self.bbox_pred(x)
    284 

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/module.py in _call_impl(self, *input, **kwargs)
   1049         if not (self._backward_hooks or self._forward_hooks or self._forward_pre_hooks or _global_backward_hooks
   1050                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1051             return forward_call(*input, **kwargs)
   1052         # Do not call functions when jit is used
   1053         full_backward_hooks, non_full_backward_hooks = [], []

~/anaconda3/lib/python3.8/site-packages/torch/nn/modules/linear.py in forward(self, input)
     94 
     95     def forward(self, input: Tensor) -> Tensor:
---> 96         return F.linear(input, self.weight, self.bias)
     97 
     98     def extra_repr(self) -> str:

~/anaconda3/lib/python3.8/site-packages/torch/nn/functional.py in linear(input, weight, bias)
   1845     if has_torch_function_variadic(input, weight):
   1846         return handle_torch_function(linear, (input, weight), input, weight, bias=bias)
-> 1847     return torch._C._nn.linear(input, weight, bias)
   1848 
   1849 

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking arugment for argument mat1 in method wrapper_addmm)
