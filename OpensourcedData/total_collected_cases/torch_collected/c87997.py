
% python -c "import torch;x=torch.rand(10);torch.save('foo', x)"
Traceback (most recent call last):
  File "/Users/nshulga/git/pytorch/pytorch/torch/serialization.py", line 423, in save
    _save(obj, opened_zipfile, pickle_module, pickle_protocol)
  File "/Users/nshulga/git/pytorch/pytorch/torch/serialization.py", line 637, in _save
    zip_file.write_record('data.pkl', data_value, len(data_value))
AttributeError: 'Tensor' object has no attribute 'write'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/nshulga/git/pytorch/pytorch/torch/serialization.py", line 424, in save
    return
  File "/Users/nshulga/git/pytorch/pytorch/torch/serialization.py", line 299, in __exit__
    self.file_like.write_end_of_file()
AttributeError: 'Tensor' object has no attribute 'write'
libc++abi: terminating with uncaught exception of type pybind11::error_already_set: AttributeError: 'Tensor' object has no attribute 'write'
zsh: abort      python -c "import torch;x=torch.rand(10);torch.save('foo', x)"
