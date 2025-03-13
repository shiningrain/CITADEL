
#import numpy as np # <- uncomment this for speedup
import torch
import matplotlib.pyplot as plt
import time

# warmup
time.sleep(1)
src_length = torch.arange(128)
for _ in range(3000):
    x = src_length.sum()
time.sleep(1)

res = []
for _ in range(3000):
    translate_timer = time.time_ns()
    x = src_length.sum()
    elapsed = time.time_ns() - translate_timer
    #print(elapsed)
    res.append(elapsed)

# save results
import numpy as np
res = np.array(res)
np.save('no_np_02', res)

# filter slow outliers to see the difference better
# threshold = 3000
# res = res[res < threshold]

print('mean {}, std {}, min {}, max {}'.format(
    np.mean(res), np.std(res), np.min(res), np.max(res)))
