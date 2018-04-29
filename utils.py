import torch


def get_broadcast_dims(input, output):
    list_dims = []
    list_not_keeps = []

    if input.dim() < output.dim():
        table = torch.zeros(input.dim(), output.dim())
        for i, v_i in enumerate(input.size()):
            for j, v_j in enumerate(output.size()):
                if v_i == v_j and all(table[i, :j] == 0):  # just accept one-to-one mapping
                    table[i, j] = 1

        for k in range(output.dim()):
            if all(table[:, k] == 0):  # add dimension here
                input.unsqueeze(k)
                list_not_keeps.append(k)

    for i, (l1, l2) in enumerate(zip(input.size(), output.size())):
        if l1 < l2:
            list_dims.append(i)

    return list_dims, set(list_not_keeps)


def check(list1, list2, vars):
    for i, (e1, e2) in enumerate(zip(list1, list2)):
        print("Compare data of variable {}:".format(vars[i]))
        if torch.equal(e1.data, e2.data):
            print('Correct')
        else:
            print('Incorrect: \n\tcomputed: {} with shape: {}\n\texpected: {} with shape: {}'
                  .format(e1.data, e1.shape, e2.data, e2.shape))

        print("Compare grad of variable {}:".format(vars[i]))
        if (e1.grad is None and e2.grad is None) or torch.equal(e1.grad.data, e2.grad.data)\
                or any(torch.isnan(e2.grad.data).tolist()):
            print('Correct')
        else:
            print('Incorrect: \n\tcomputed: {} with shape: {}\n\texpected: {} with shape: {}'
                  .format(e1.grad.data, e1.grad.shape, e2.grad.data, e2.grad.shape))

        print('---')
