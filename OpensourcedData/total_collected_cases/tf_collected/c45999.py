TESTING 2 x 2 INPUT WITH 2 x 2 KERNEL, (1, 1) STRIDE AND "SAME" PADDING:
Input:
[[1 1]
 [1 1]]

Kernel:
[[1 1]
 [1 1]]


Dilated Input: (=> 2 x 2 array)
[[1 1]
 [1 1]]

Dilated Input + Padding: (=> 3 x 3 array)
[[1 1 0]
 [1 1 0]
 [0 0 0]]


Manual Result: (=> 2 x 2 array)
[[4 2]
 [2 1]]

Standard Result: (=> 2 x 2 array)
[[1 2]
 [2 4]]


Here's an example with an odd kernel and valid padding instead:
Python
TESTING 2 x 2 INPUT WITH 3 x 3 KERNEL, (5, 5) STRIDE AND "VALID" PADDING:
Input:
[[1 1]
 [1 1]]

Kernel:
[[1 1 1]
 [1 1 1]
 [1 1 1]]


Dilated Input: (=> 6 x 6 array)
[[1 0 0 0 0 1]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]
 [0 0 0 0 0 0]
 [1 0 0 0 0 1]]

Dilated Input + Padding: (=> 12 x 12 array)
[[0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]


Manual Result: (=> 10 x 10 array)
[[0 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 0 0 1 1 1 0]
 [0 1 1 1 0 0 1 1 1 0]
 [0 1 1 1 0 0 1 1 1 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 0 0 1 1 1 0]
 [0 1 1 1 0 0 1 1 1 0]
 [0 1 1 1 0 0 1 1 1 0]
 [0 0 0 0 0 0 0 0 0 0]]

Standard Result: (=> 10 x 10 array)
[[1 1 1 0 0 1 1 1 0 0]
 [1 1 1 0 0 1 1 1 0 0]
 [1 1 1 0 0 1 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [1 1 1 0 0 1 1 1 0 0]
 [1 1 1 0 0 1 1 1 0 0]
 [1 1 1 0 0 1 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]]


Here's an example with an odd signal and even kernel:
Python
TESTING 3 x 3 INPUT WITH 2 x 2 KERNEL, (4, 4) STRIDE AND "VALID" PADDING:
Input:
[[1 1 1]
 [1 1 1]
 [1 1 1]]

Kernel:
[[1 1]
 [1 1]]


Dilated Input: (=> 9 x 9 array)
[[1 0 0 0 1 0 0 0 1]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [1 0 0 0 1 0 0 0 1]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0]
 [1 0 0 0 1 0 0 0 1]]

Dilated Input + Padding: (=> 13 x 13 array)
[[0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 1 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 1 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 0 1 0 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]]


Manual Result: (=> 12 x 12 array)
[[0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 1 0 0 1 1 0 0 1 1 0]
 [0 1 1 0 0 1 1 0 0 1 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 1 0 0 1 1 0 0 1 1 0]
 [0 1 1 0 0 1 1 0 0 1 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 1 1 0 0 1 1 0 0 1 1 0]
 [0 1 1 0 0 1 1 0 0 1 1 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]

Standard Result: (=> 12 x 12 array)
[[1 1 0 0 1 1 0 0 1 1 0 0]
 [1 1 0 0 1 1 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [1 1 0 0 1 1 0 0 1 1 0 0]
 [1 1 0 0 1 1 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [1 1 0 0 1 1 0 0 1 1 0 0]
 [1 1 0 0 1 1 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]


Finally, here the skew is quite large and noticeable due to the large amount of padding, which clearly seems like it could only have stemmed from the padding being allocated entirely to the bottom and right only:
Python
TESTING 2 x 2 INPUT WITH 2 x 2 KERNEL, (6, 6) STRIDE AND "SAME" PADDING:
Input:
[[1 1]
 [1 1]]

Kernel:
[[1 1]
 [1 1]]


Dilated Input: (=> 7 x 7 array)
[[1 0 0 0 0 0 1]
 [0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0]
 [1 0 0 0 0 0 1]]

Dilated Input + Padding: (=> 13 x 13 array)
[[0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0 0 1 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0]]


Manual Result: (=> 12 x 12 array)
[[0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 0 0 0 1 1 0 0]
 [0 0 1 1 0 0 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 1 0 0 0 0 1 1 0 0]
 [0 0 1 1 0 0 0 0 1 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]

Standard Result: (=> 12 x 12 array)
[[1 1 0 0 0 0 1 1 0 0 0 0]
 [1 1 0 0 0 0 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [1 1 0 0 0 0 1 1 0 0 0 0]
 [1 1 0 0 0 0 1 1 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0]]


**Describe the expected behavior**
Padding for transpose convolutions should be distributed evenly with at most an extra zero on the right when not evenly divisible by 2. This is TensorFlow convention for regular convolutions, but not here, oddly enough.

All of my examples are easy to compute and verify by hand. It is not clear what the exact method of allocating the padding is for transpose convolutions, as the documentation is quite scarce. Just to clarify, my code does actually match up for many parameter combinations I did not show here for brevity, but it disagrees with enough to make me suspicious that something's up. I would love it if someone could explain the algorithm better!

**Standalone code to reproduce the issue**
I have provided code to replicate the above examples below:
Python
# Import modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose

# Define convenience functions
def dilate_array(x, kernel, dilation_rate, padding = 'valid'): # manually dilate array by inserting zeros
    # Get padding for after dilation
    rank = len(x)
    x = x[None, ..., None]
    paddings, _ = get_deconv_padding(x, kernel, strides, padding = padding) # calculate BEFORE dilation
    
    # Create "upsampled" array via repetition then cut off extra values by slicing
    shp, strd, slc = (1, ), (slice(None, None, None), ), (slice(None, None, None), )
    dil = inp = Input(x.shape[1:])
    for i in range(rank):
        shp += (dilation_rate[i]*dil.shape[i + 1] - dilation_rate[i] + 1, )
        dil = tf.repeat(dil, dilation_rate[i], axis = i + 1)
        strd += (slice(None, None, dilation_rate[i]), )
        slc += (slice(None, shp[i + 1], None), )
    shp += (1, )
    strd += (slice(None, None, None), )
    slc += (slice(None, None, None), )
    dil = dil[slc]
    
    # Create mask to replace repeats with zeros a la "proper" resampling and transpose convolution
    mask = np.zeros(shp)
    mask[strd] = 1.0
    mask = tf.cast(mask, dil.dtype)
    
    # Mask dilated input
    dil = mask*dil
    
    # Pad dilated input
    pad = tf.pad(dil, paddings, mode = 'CONSTANT')
    
    # Execute dilation + padding
    model = Model(inp, [dil, pad])
    xd, xp = model.predict(x)
    xd, xp = xd[0, ..., 0], xp[0, ..., 0]
    
    return xd, xp, paddings
def deconv_padding_after_dilation(input_length, kernel_length, stride, padding = 'valid'): # adjusted padding for manual dilation (== stride)
    # Compute minimum padding length assuming manually dilated array
    output_length = stride*input_length + (max(kernel_length - stride, 0) if padding == 'valid' else 0)
    dilated_length = stride*input_length - (stride - 1)
    pad_length = output_length - dilated_length + kernel_length - 1
    
    return pad_length, output_length
def get_deconv_padding(x, kernel, strides, padding = 'valid'):
    # Calculate needed paddings with PRE-DILATED input to apply to DILATED input
    rank = len(x.shape) - 2
    paddings, output_shape = [[0, 0]], (1, )
    for i in range(rank):
        pad_length, output_length = deconv_padding_after_dilation(int(x.shape[i + 1]), kernel.shape[i], strides[i], padding = padding)
        paddings += [[pad_length//2, pad_length//2 + pad_length%2]]
        output_shape += (output_length, )
    paddings += [[0, 0]]
    output_shape += (1, )
    
    return paddings, output_shape
def execute_manual_transpose_convolution(x, kernel, strides, padding = 'valid'):
    # Manually dilate and pad output then convolve normally
    xd, xp, paddings = dilate_array(x, kernel, strides, padding = padding)
    inp = Input(xp.shape + (1, ))
    conv = Conv2D(1, kernel.shape, padding = 'valid')(inp)
    model = Model(inp, conv)
    model.set_weights([kernel[..., None, None], np.zeros((1, ))])
    y = model.predict(xp[None, ..., None])[0, ..., 0]
    
    return xd, xp, paddings, y
def execute_standard_transpose_convolution(x, kernel, strides, padding = 'valid'):
    # Use standard transpose convolution
    inp = Input(x.shape + (1, ))
    convT = Conv2DTranspose(1, kernel.shape, strides = strides, padding = padding)(inp)
    model = Model(inp, convT)
    model.set_weights([kernel[..., None, None], np.zeros((1, ))])
    y = model.predict(x[None, ..., None])[0, ..., 0]
    
    return y

