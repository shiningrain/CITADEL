
import torch.utils.cpp_extension

cpp_src = """

torch::Tensor a(const torch::Tensor& self) {
    return self;
}

torch::Tensor my_test_sparse_op(const torch::Tensor& self, const torch::Tensor& other) {
    auto values = self._values();
    auto indices = self._indices();
    std::cout << (values.unsafeGetTensorImpl() == self._values().unsafeGetTensorImpl()) << std::endl; // false
    values.resize_as_(other._values());
    values.zero_();
    indices.resize_as_(other._indices());
    indices.copy_(other._indices());
    return self;
}

TORCH_LIBRARY(my_ops, m) {
  m.def("my_test_sparse_op(Tensor self, Tensor other) -> Tensor ");
}

TORCH_LIBRARY_IMPL(my_ops, SparseCPU, m) {
  m.impl("my_test_sparse_op", my_test_sparse_op);
}
"""

torch.utils.cpp_extension.load_inline(
    name="my_ops",
    cpp_sources=cpp_src,
    is_python_module=False,
    verbose=True,
)

indices = torch.zeros((1, 0))
values = torch.zeros((0,))
s1 = torch.sparse_coo_tensor(indices=indices, values=values, size=(3,)).coalesce()
print(s1)

indices = torch.tensor([0, 1, 2], dtype=torch.int64).reshape((1, 3))
values = torch.ones((3,))
s2 = torch.sparse_coo_tensor(indices=indices, values=values, size=(3,)).coalesce()
print(s2)

c = torch.ops.my_ops.my_test_sparse_op(s1, s2)

print(c) // expect c to have same indices as s2, but 
