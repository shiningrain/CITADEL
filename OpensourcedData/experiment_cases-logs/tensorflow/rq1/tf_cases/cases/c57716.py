import numpy as np
import sys
import pickle
sys.path.append('../../tf_code')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
import tensorflow as tf
func_cls(data=np.ones((3)),segment_ids=898042203, num_segments=8327099846119777499)



# import numpy as np
# import tensorflow as tf
# tf.math.unsorted_segment_max(data=np.ones((3)),segment_ids=898042203, num_segments=8327099846119777499)



# import numpy as np
# import tensorflow as tf
# tf.math.unsorted_segment_sum(data=np.ones((3)),segment_ids=898042203, num_segments=8327099846119777499)



# import numpy as np
# import tensorflow as tf
# tf.math.unsorted_segment_prod(data=np.ones((3)),segment_ids=898042203, num_segments=8327099846119777499)

