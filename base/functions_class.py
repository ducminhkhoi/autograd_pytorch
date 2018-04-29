from base.base_function import BinaryFunction, UnaryFunction
import torch
from utils import get_broadcast_dims


class ReLU(UnaryFunction):
    op = 'ReLU'

    def __init__(self):
        super().__init__()

    def forward_(self):
        pos_mask = self.input > 0
        self.context['pos_mask'] = pos_mask
        return self.input[pos_mask]

    def backward_(self):
        pos_mask = self.context['pos_mask']
        out = torch.zeros_like(pos_mask)
        out[pos_mask] = 1
        return out


# ------- Reduction Function --------- #

class Sum(UnaryFunction):
    op = 'sum'

    def backward_(self):
        dim, keepdim = self.get_args(['dim', 'keepdim'])
        self.grad = torch.ones_like(self.input)


class Max(UnaryFunction):
    op = 'max'

    def backward_(self):
        self.get_args(['dim', 'keepdim'])

        self.grad = torch.zeros_like(self.input).float()

        if isinstance(self.output, (tuple, list)):
            index = self.output[1]
            self.grad[index] = 1.
        else:  # max of all elements
            index = getattr(self.input.view(-1, 1), self.op)(0)[1]
            self.grad = self.grad.view(-1, 1)
            self.grad[index] = 1.
            self.grad = self.grad.view(*self.input.size())


class Min(Max):
    op = 'min'


class Prod(UnaryFunction):
    op = 'prod'

    def backward_(self):
        dim, keepdim = self.get_args(['dim', 'keepdim'])
        if dim is not None and not keepdim:
            self.grad = self.output.unsqueeze(dim).expand_as(self.input) / self.input
        else:
            self.grad = self.output.expand_as(self.input) / self.input


class Mean(UnaryFunction):
    op = 'mean'

    def backward_(self):
        dim, keepdim = self.get_args(['dim', 'keepdim'])

        if dim is not None:
            n = self.input.size(dim)
        else:
            n = self.input.numel()

        self.grad = torch.ones_like(self.input) / n


class Var(UnaryFunction):
    op = 'var'

    def backward_(self):
        dim, keepdim, unbiased = self.get_args(['dim', 'keepdim', 'unbiased'])

        if dim is not None:
            n = self.input.size(dim)
        else:
            n = self.input.numel()

        mean_grad = torch.ones_like(self.input) / n
        temp_kwargs = dict(self.kwargs)
        temp_kwargs.pop('unbiased', None)

        mean = self.input.mean(*self.args, **temp_kwargs)
        if dim is not None:
            mean = mean.unsqueeze(dim)

        temp_grad = 2 * (self.input - mean) * (1 - mean_grad)

        if unbiased:  # use n - 1, else
            self.grad = temp_grad / (n - 1)
        else:  # use n
            self.grad = temp_grad / n


class Norm(UnaryFunction):
    op = 'norm'

    def backward_(self):
        p, dim, keepdim = self.get_args(['p', 'dim', 'keepdim'])

        if p is None:
            p = 2

        if dim is None:
            if p == 2:
                self.grad = self.input / self.output
            else:
                pow = self.input.abs().pow(p - 2)
                scale = 1. / self.output ** (p - 1)
                self.grad = self.input * pow * scale
        else:
            if keepdim is False:
                self.in_grad = self.in_grad.unsqueeze(dim)
                self.output = self.output.unsqueeze(dim)

            self.in_grad = self.in_grad.expand_as(self.input)
            if p == 2:
                big_output = self.output.expand_as(self.input)
                self.grad = self.input / big_output
            else:
                pow = self.input.abs().pow(p - 2)
                big_output = self.output.pow(p - 1).expand_as(self.input)
                self.grad = self.input * pow / big_output


class T(UnaryFunction):
    op = 't'

    def backward_(self):
        self.grad = torch.ones_like(self.input)
        self.in_grad = self.in_grad.t()


class Transpose(UnaryFunction):
    op = 'transpose'

    def backward_(self):
        self.grad = torch.ones_like(self.input)
        self.in_grad = self.in_grad.transpose(*self.args, **self.kwargs)


class Permute(UnaryFunction):
    op = 'permute'

    def forward_(self):
        self.output = self.input.permute(*self.args,  **self.kwargs)

    def backward_(self):
        self.grad = torch.ones_like(self.input)
        self.in_grad = self.in_grad.permute(*self.args,  **self.kwargs)


class Exp(UnaryFunction):
    op = 'exp'

    def backward_(self):
        self.grad = self.output


class Sqrt(UnaryFunction):
    op = 'sqrt'

    def backward_(self):
        self.grad = 1 / (2 * self.output)


class Abs(UnaryFunction):
    op = 'abs'

    def backward_(self):
        self.grad = self.input.sign()


class Sin(UnaryFunction):
    op = 'sin'

    def backward_(self):
        self.grad = self.input.cos()


class Cos(UnaryFunction):
    op = 'cos'

    def backward_(self):
        self.grad = -self.input.sin()


class Tan(UnaryFunction):
    op = 'tan'

    def backward_(self):
        self.grad = 1 + self.input.tan()**2


class Tanh(UnaryFunction):
    op = 'tanh'

    def backward_(self):
        self.grad = 1 - self.input.tan()**2


class Sigmoid(UnaryFunction):
    op = 'sigmoid'

    def backward_(self):
        self.grad = self.output * (1 - self.output)


class Log(UnaryFunction):
    op = 'log'

    def backward_(self):
        self.grad = torch.reciprocal(self.input)


class Add(BinaryFunction):
    op = 'add'

    def backward_(self):
        self.grad1 = torch.ones_like(self.input1)
        self.grad2 = torch.ones_like(self.input2)


class Mul(BinaryFunction):
    op = 'mul'

    def backward_(self):
        self.grad1 = self.input2
        self.grad2 = self.input1


class MatMul(BinaryFunction):
    op = 'matmul'

    def backward_(self):
        self.grad1 = self.input2
        self.grad2 = self.input1


class Pow(BinaryFunction):
    op = 'pow'

    def backward_(self):
        self.grad1 = self.input2 * self.input1 ** (self.input2 - 1)
        self.grad2 = torch.log(self.input1) * self.input1 ** self.input2


class Max2(BinaryFunction):
    op = 'max'

    def backward_(self):
        input1 = self.input1
        input2 = self.input2

        if input1.numel() > input2.numel():  # use broadcast
            input2 = input2.expand_as(input1).contiguous()
        else:
            input1 = input1.expand_as(input2).contiguous()

        input1 = input1.view(-1)
        input2 = input2.view(-1)
        input = torch.stack([input1, input2], 1)

        grad = torch.zeros_like(input).float()
        index = getattr(input, self.op)(1)[1]

        grad[range(len(index)), index] = 1.

        grad1, grad2 = grad[:, 0], grad[:, 1]

        if grad1.numel() != self.input1.numel():  # input1 is broadcasted
            grad1 = grad1.view_as(self.input2)

            list_dims, list_not_keeps = get_broadcast_dims(self.input1, grad1)

            for dim in list_dims:
                if dim in list_not_keeps:
                    grad1 = grad1.sum(dim, keepdim=False) / grad1.size(dim)
                else:
                    grad1 = grad1.sum(dim, keepdim=True) / grad1.size(dim)

        self.grad1 = grad1.view_as(self.input1)

        if grad2.numel() != self.input2.numel():
            grad2 = grad2.view_as(self.input1)

            list_dims, list_not_keeps = get_broadcast_dims(self.input2, grad2)

            for dim in list_dims:
                if dim in list_not_keeps:
                    grad2 = grad2.sum(dim, keepdim=False) / grad2.size(dim)
                else:
                    grad2 = grad2.sum(dim, keepdim=True) / grad2.size(dim)

        self.grad2 = grad2.view_as(self.input2)


class Min2(Max2):
    op = 'min'


class Cat(UnaryFunction):
    op = 'cat'

    def __init__(self):
        super().__init__()
        self.input = None
        self.grad = None
        self.args = None
        self.kwargs = None

    def forward(self, input, *args, **kwargs):
        self.input = input
        self.args = args
        self.kwargs = kwargs
        self.output = getattr(torch, self.op)([x.data for x in self.input], *args, **kwargs)

        return self.output

    def backward_(self):
        self.grad = torch.ones_like(self.in_grad)

    def backward__(self):
        if self.input[0].data.is_cuda:
            self.grad = self.grad.cuda()

        dim, = self.get_args(['dim'])

        if dim is None:
            dim = 0

        list_lengths = [x.size(dim) for x in self.input]

        out_grad = self.in_grad * self.grad

        out_grads = out_grad.split(list_lengths, dim)

        for input_, grad in zip(self.input, out_grads):
            input_.backward(grad)


class Stack(Cat):
    op = 'stack'


class Squeeze(UnaryFunction):
    op = 'squeeze'

    def backward_(self):
        self.grad = torch.ones_like(self.input)
        dim, = self.get_args(['dim'])

        if dim is not None:
            self.in_grad = self.in_grad.unsqueeze(*self.args, **self.kwargs)
        else:
            squeeze_dims = [i for i, x in enumerate(self.input.size()) if x == 1]
            for dim in squeeze_dims:
                self.in_grad = self.in_grad.unsqueeze(dim)


class Unsqueeze(UnaryFunction):
    op = 'unsqueeze'

    def backward_(self):
        self.grad = torch.ones_like(self.input)
        self.in_grad = torch.squeeze(self.in_grad.data, *self.args, **self.kwargs)


class Split(UnaryFunction):
    op = 'split'

    def backward_(self):
        self.grad = torch.ones_like(self.input)

    def backward(self, in_grad):
        dim, = self.get_args(['dim'])

        if dim is None:
            dim = 0

        self.in_grad = torch.cat([x.data for x in in_grad], dim)
        self.backward_()
        self.backward__()


class Getitem(UnaryFunction):
    op = '__getitem__'

    def forward_(self):
        self.output = self.input.__getitem__(*self.args, **self.kwargs)

    def backward_(self):
        self.grad = torch.zeros_like(self.input)


class View(UnaryFunction):
    op = 'view'

    def forward_(self):
        self.output = self.input.view(*self.args, **self.kwargs)

    def backward_(self):
        self.grad = torch.ones_like(self.input)
        self.in_grad = self.in_grad.view(*self.input.size())
