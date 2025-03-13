
import torch
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
arg_1 = torch.rand([5, 5], dtype=torch.float64)
arg_2 = torch.rand([5, 5], dtype=torch.float64)
arg_3 = torch.rand([1, 5], dtype=torch.complex128)
res = func_cls(arg_1,arg_2,arg_3)
# RuntimeError: !(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ && (has_undefined_outputs || config.enforce_safe_casting_to_output_ || config.cast_common_dtype_to_outputs_))INTERNAL ASSERT FAILED at "../aten/src/ATen/TensorIterator.cpp":331, please report a bug to PyTorch. 
