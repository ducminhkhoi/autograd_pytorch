import torch
from utils import check


def check_example(pytorch=False):

    if pytorch:
        from torch.autograd import Variable
    else:
        from variables import Variable

    a = Variable(torch.Tensor([[-4, 2], [8, 4], [5, 6]]).float(), requires_grad=True)
    b = Variable(torch.Tensor([[2, 3, 4], [5, 6, 7]]).float(), requires_grad=True)
    c = Variable(torch.Tensor([[4, 5, 6]]).float(), requires_grad=True)

    if pytorch:
        import torch as F
    else:
        import functions as F

    d = (a * b.permute(1, 0)).view(2, 3)
    g = d.sum()

    # k = a + 1
    # d = (k @ b).tanh()
    # e = d @ c.t()
    # f = e ** a.sum(1)
    # g = f.norm(dim=0, keepdim=True)
    # g = g.sum()

    # k.retain_grad()
    # c.retain_grad()
    d.retain_grad()
    # e.retain_grad()
    # f.retain_grad()
    g.retain_grad()

    g.backward()

    local = locals()
    vars = list('abcdg')
    ret_tuple = tuple(local[x] for x in vars)
    return ret_tuple, vars


if __name__ == '__main__':
    ret1, vars = check_example()

    ret2 = check_example(pytorch=True)[0]

    check(ret1, ret2, vars)