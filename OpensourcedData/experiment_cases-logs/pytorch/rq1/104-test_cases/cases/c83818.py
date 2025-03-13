
import torch as t
import sys
import pickle
sys.path.append('../../codes')
from run_utils import string2function
func_cls=string2function(sys.argv[1])
func_cls(t.randn([2895, 2895]))

