import torch
import functions as F


class Variable:
    def __init__(self, data, grad_fn=None, requires_grad=None):
        self.data = data
        self.grad = None
        self.grad_fn = grad_fn

    def __matmul__(self, other):
        return F.matmul(self, other)

    def __mul__(self, other):
        return F.mul(self, other)

    def __add__(self, other):
        return F.add(self, other)

    def __sub__(self, other):
        return F.sub(self, other)

    def __rsub__(self, other):
        return F.sub(other, self)

    def __truediv__(self, other):
        return F.truediv(self, other)

    def __rtruediv__(self, other):
        return F.truediv(other, self)

    def __pow__(self, power, modulo=None):
        return F.pow(self, power)

    def __rpow__(self, other):
        return F.pow(other, self)

    def __abs__(self):
        return F.abs(self)

    def __neg__(self):
        return F.neg(self)

    def __getitem__(self, *args, **kwargs):
        return F.getitem(self, *args, **kwargs)

    def size(self, *args):
        return self.data.size(*args)

    @property
    def shape(self):
        return self.data.size()

    @property
    def T(self):
        return self.t()

    def t(self):
        return F.t(self)

    def transpose(self, *args, **kwargs):
        return F.transpose(self, *args, **kwargs)

    def permute(self, *args, **kwargs):
        return F.permute(self, *args, **kwargs)

    def split(self, *args, **kwargs):
        return F.split(self, *args, **kwargs)

    def squeeze(self, *args, **kwargs):
        return F.squeeze(self, *args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return F.unsqueeze(self, *args, **kwargs)

    def view(self, *args, **kwargs):
        return F.view(self, *args, **kwargs)

    def dim(self):
        return self.data.dim()

    def numel(self):
        return self.data.numel()

    def abs(self):
        return F.abs(self)

    def sin(self):
        return F.sin(self)

    def cos(self):
        return F.cos(self)

    def tan(self):
        return F.tan(self)

    def tanh(self):
        return F.tanh(self)

    def exp(self):
        return F.exp(self)

    def log(self):
        return F.log(self)

    def sqrt(self):
        return F.sqrt(self)

    def sigmoid(self):
        return F.sigmoid(self)

    # Reduction op

    def sum(self, *args, **kwargs):
        return F.sum(self, *args, **kwargs)

    def norm(self, *args, **kwargs):
        return F.norm(self, *args, **kwargs)

    def mean(self, *args, **kwargs):
        return F.mean(self, *args, **kwargs)

    def var(self, *args, **kwargs):
        return F.var(self, *args, **kwargs)

    def std(self, *args, **kwargs):
        return F.std(self, *args, **kwargs)

    def max(self, *args, **kwargs):
        return F.max(self, *args, **kwargs)

    def min(self, *args, **kwargs):
        return F.min(self, *args, **kwargs)

    def prod(self, *args, **kwargs):
        return F.prod(self, *args, **kwargs)

    __radd__ = __add__
    __rmul__ = __mul__
    __iadd__ = __add__

    def retain_grad(self):
        pass

    def backward(self, in_grad=None):

        if in_grad is None:
            if self.data.size() != torch.Size([]):
                raise RuntimeError('grad can be implicitly created only for scalar outputs')
            temp_grad = Variable(torch.tensor(1.).cuda() if self.data.is_cuda else torch.tensor(1.))
        else:
            temp_grad = Variable(in_grad)

        self.grad = (self.grad if self.grad is not None else 0) + temp_grad

        if self.grad_fn:
            self.grad_fn.backward(self.grad)

    def __repr__(self):
        return 'Variable containing: {}'.format(self.data.__repr__())


class Parameter(Variable):
    def __init__(self, tensor):
        super().__init__(tensor)

    def uniform_(self, a, b):
        self.data = torch.rand(*self.data.size()) * (b - a) + a
