from base.functions_class import *


def binary_op(op1, op2, op_):
    if op1.__class__ == op2.__class__:
        Variable = op2.__class__
    else:
        if isinstance(op1, (float, int, bool)):
            Variable = op2.__class__
            op1 = Variable(torch.tensor(float(op1)))
        else:
            Variable = op1.__class__
            op2 = Variable(torch.tensor(float(op2)))

    op = op_()
    out = op(op1, op2)
    return Variable(out, op)


def unary_op(op1, op_, *args, **kwargs):
    if not isinstance(op1, (tuple, list)):
        Variable = op1.__class__
    else:
        Variable = op1[0].__class__

    op = op_()
    out = op(op1, *args, **kwargs)

    if isinstance(out, torch.Tensor):
        return Variable(out, op)
    elif isinstance(out, (tuple, list)):  # return 2 values
        return [Variable(o, op) for o in out]


def add(op1, op2):
    return binary_op(op1, op2, Add)


def matmul(op1, op2):
    return binary_op(op1, op2, MatMul)


def mul(op1, op2):
    return binary_op(op1, op2, Mul)


def add(op1, op2):
    return binary_op(op1, op2, Add)


def sub(op1, op2):
    return binary_op(op1, -1 * op2, Add)


def truediv(op1, op2):
    return binary_op(op1, op2 ** -1, Mul)


def pow(op1, power, modulo=None):
    return binary_op(op1, power, Pow)


def t(op):
    return unary_op(op, T)


def transpose(op, *args, **kwargs):
    return unary_op(op, Transpose, *args, **kwargs)


def permute(op, *args, **kwargs):
    return unary_op(op, Permute, *args, **kwargs)


def abs(op):
    return unary_op(op, Abs)


def sin(op):
    return unary_op(op, Sin)


def cos(op):
    return unary_op(op, Cos)


def tan(op):
    return unary_op(op, Tan)


def tanh(op):
    return unary_op(op, Tanh)


def neg(op):
    return binary_op(op, -1, Mul)


def sum(op, *args, **kwargs):
    return unary_op(op, Sum, *args, **kwargs)


def exp(op):
    return unary_op(op, Exp)


def log(op):
    return unary_op(op, Log)


def sqrt(op):
    return unary_op(op, Sqrt)


def sigmoid(op):
    return unary_op(op, Sigmoid)


def norm(op, *args,  **kwargs):
    return unary_op(op, Norm, *args, **kwargs)


def mean(op1, *args, **kwargs):
    return unary_op(op1, Mean, *args, **kwargs)


def prod(op1, *args, **kwargs):
    return unary_op(op1, Prod, *args, **kwargs)


def var(op1, *args, **kwargs):
    return unary_op(op1, Var, *args, **kwargs)


def std(op1, *args, **kwargs):
    return unary_op(op1, Var, *args, **kwargs).sqrt()


def max(op1, *args, **kwargs):
    if isinstance(args[0], op1.__class__):
        return binary_op(op1, args[0], Max2)
    else:
        return unary_op(op1, Max, *args, **kwargs)


def min(op1, *args, **kwargs):
    if isinstance(args[0], op1.__class__):
        return binary_op(op1, args[0], Min2)
    else:
        return unary_op(op1, Min, *args, **kwargs)


def cat(list_ops, *args, **kwargs):
    return unary_op(list_ops, Cat, *args, **kwargs)


def stack(list_ops, *args, **kwargs):
    return unary_op(list_ops, Stack, *args, **kwargs)


def squeeze(op1, *args, **kwargs):
    return unary_op(op1, Squeeze, *args, **kwargs)


def unsqueeze(op1, *args, **kwargs):
    return unary_op(op1, Unsqueeze, *args, **kwargs)


def split(op1, *args, **kwargs):
    return unary_op(op1, Split, *args, **kwargs)


def getitem(op1, *args, **kwargs):
    return unary_op(op1, Getitem, *args, **kwargs)


def view(op1, *args, **kwargs):
    return unary_op(op1, View, *args, **kwargs)

