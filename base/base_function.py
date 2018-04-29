import torch
from utils import get_broadcast_dims


class Function:
    def __init__(self):
        self.context = dict()
        self.in_grad = None
        self.output = None

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *tensors):
        raise NotImplementedError

    def forward_(self):
        raise NotImplementedError

    def backward_(self):
        raise NotImplementedError

    def backward__(self):
        raise NotImplementedError

    def backward(self, in_grad):
        self.in_grad = in_grad.data
        self.backward_()
        self.backward__()

    def get_args(self, list_options):

        list_args = []

        for i, option in enumerate(list_options):
            var = None
            if self.args is not None:
                if len(self.args) > i:
                    var = self.args[i]

            if self.kwargs is not None:
                if option in self.kwargs:
                    var = self.kwargs[option]

            self.context[option] = var

            list_args.append(var)

        return list_args


class BinaryFunction(Function):
    def __init__(self):
        super().__init__()
        self.input1 = None
        self.input2 = None
        self.input1_var = None
        self.input2_var = None
        self.grad1 = None
        self.grad2 = None
        self.args = None
        self.kwargs = None

    def forward(self, input1, input2, *args, **kwargs):
        self.input1 = input1.data
        self.input2 = input2.data
        self.input1_var = input1
        self.input2_var = input2
        self.args = args
        self.kwargs = kwargs
        self.forward_()
        return self.output

    def forward_(self):
        self.output = getattr(torch, self.op)(self.input1, self.input2, *self.args, **self.kwargs)

    def backward__(self):
        if self.input1.data.is_cuda:
            self.grad1 = self.grad1.cuda()

        if self.input2.data.is_cuda:
            self.grad2 = self.grad2.cuda()

        if self.op == 'matmul':
            out_grad1 = self.in_grad @ self.grad1.t()
            out_grad2 = self.grad2.t() @ self.in_grad
        else:
            out_grad1 = self.in_grad * self.grad1
            out_grad2 = self.in_grad * self.grad2

        if self.input1.data.dim() != 0:  # not a scalar value, it needs gradient
            if out_grad1.shape != self.input1.shape:  # apply broadcast in forward pass
                list_dims, list_not_keeps = get_broadcast_dims(self.input1, out_grad1)

                for dim in list_dims:
                    if dim in list_not_keeps:
                        out_grad1 = out_grad1.sum(dim, keepdim=False)
                    else:
                        out_grad1 = out_grad1.sum(dim, keepdim=True)

                out_grad1 = out_grad1.view_as(self.input1)

            self.input1_var.backward(out_grad1)

        if self.input2.data.dim() != 0:  # not a scalar value, it needs gradient
            if out_grad2.shape != self.input2.shape:  # apply broadcast in forward pass
                list_dims, list_not_keeps = get_broadcast_dims(self.input2, out_grad2)

                for dim in list_dims:
                    if dim in list_not_keeps:
                        out_grad2 = out_grad2.sum(dim, keepdim=False)
                    else:
                        out_grad2 = out_grad2.sum(dim, keepdim=True)

                out_grad2 = out_grad2.view_as(self.input2)

            self.input2_var.backward(out_grad2)


class UnaryFunction(Function):
    def __init__(self):
        super().__init__()
        self.input = None
        self.input_var = None
        self.grad = None
        self.args = None
        self.kwargs = None

    def forward_(self):
        self.output = getattr(torch, self.op)(self.input, *self.args, **self.kwargs)

    def forward(self, input, *args, **kwargs):
        self.input = input.data
        self.input_var = input
        self.args = args
        self.kwargs = kwargs
        self.forward_()

        return self.output

    def backward__(self):
        if self.input.data.is_cuda:
            self.grad = self.grad.cuda()

        if self.input.numel() != self.in_grad.numel():  # reduce op
            if self.op == 'getitem':
                self.grad.__setitem__(*self.args, 1, **self.kwargs)
            else:
                dim = self.context['dim']
                if dim is not None:
                    self.in_grad = self.in_grad.unsqueeze(dim)
                else:
                    self.in_grad = self.in_grad.unsqueeze(-1)

        out_grad = self.in_grad * self.grad

        self.input_var.backward(out_grad)
