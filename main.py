# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import torch
import time
import math
import os


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def ackley(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    a = 20
    b = 0.2
    c = 2 * math.pi

    x = rescale(x, xmin, xmax, -32.768, 32.768)

    term1 = x.pow(2).mean(1).sqrt().mul(-b).exp().mul(-a)
    term2 = x.mul(c).cos().mean(1).exp().mul(-1)

    return term1 + term2 + a + math.exp(1)


def michal(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/michal.html
    """
    x = rescale(x, xmin, xmax, 0, math.pi)

    mask = torch.arange(1, x.size(1) + 1).view(1, -1)

    return x.sin().mul((x.pow(2) * mask / math.pi).sin().pow(20)).sum(1).mul(-1)


def schwefel(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)

    result = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))

    return result


def sphere(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/spheref.html
    """
    x = rescale(x, xmin, xmax, -5.12, 5.12)

    return x.pow(2).mean(1)


def tang(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/stybtang.html
    """
    x = rescale(x, xmin, xmax, -5, 5)

    result = (x.pow(4) - x.pow(2).mul(16) + x.mul(5)).sum(1).div(2)

    return result


def rastrigin(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/rastr.html
    """
    x = rescale(x, xmin, xmax, -5.12, 5.12)

    result = (x.pow(2.0) - x.mul(2.0 * math.pi).cos().mul(10)
              ).sum(1).add(10 * x.size(1))

    return result


def rosenbrock(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/rosen.html
    """
    x = rescale(x, xmin, xmax, -5, 10)

    term1 = (x[:, 1:] - x[:, :-1].pow(2)).pow(2).mul(100).sum(1)
    term2 = x[:, :-1].add(-1).pow(2).sum(1)

    return term1 + term2


def levy(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/levy.html
    """
    x = rescale(x, xmin, xmax, -10, 10)

    w = 1 + (x - 1).div(4)
    w1 = w[:, 0]
    ww = w[:, :x.size(1) - 1]
    wd = w[:, x.size(1) - 1]

    if ww.dim() == 1:
        ww = ww.view(-1, 1)

    term1 = w1.mul(math.pi).sin().pow(2)
    termw = ww.mul(math.pi).add(1).sin().pow(2).mul(
        10).add(1).mul(ww.add(-1).pow(2)).mean(1)
    termd = wd.mul(math.pi * 2).sin().pow(2).add(1).mul(wd.add(-1).pow(2))

    return term1 + termw + termd


def energy(d):
    """
    https://en.wikipedia.org/wiki/Spin_glass
    """
    coeff = torch.randn(d, d, d)

    def foo(x, xmin=-1, xmax=1):
        x = rescale(x, xmin, xmax, -1, 1)
        x_norm = x.norm(2, 1).view(-1, 1) + 1e-10
        x = x / x_norm * math.sqrt(x.size(1))
        bs = x.size(0)
        tmp1 = torch.mul(x.view(bs, d, 1), x.view(bs, 1, d))
        tmp2 = torch.mul(tmp1.view(bs, d * d, 1), x.view(bs, 1, d))
        tmp3 = torch.mul(tmp2.view(bs, d * d * d), coeff.view(1, -1))
        return tmp3.sum(1) / d

    return foo


test_functions = {
    "ackley": ackley,
    "michal": michal,
    "schwefel": schwefel,
    "sphere": sphere,
    "tang": tang,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "levy": levy
}


class ScaledTanh(torch.nn.Module):
    def __init__(self, a=1):
        super(ScaledTanh, self).__init__()
        self.a = a

    def forward(self, x):
        return x.mul(self.a).tanh() 


class EmbeddingPerceptron(torch.nn.Module):
    """
    Multilayer ReLU perceptron with learnable inputs
    """

    def __init__(self, sizes, multiplier=3):
        super(EmbeddingPerceptron, self).__init__()
        self.inputs = torch.arange(0, sizes[0]).long()

        layers = [torch.nn.Embedding(sizes[0], sizes[1])]
        for i in range(1, len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(torch.nn.ReLU())
            
        self.net = torch.nn.Sequential(*layers)

        net_min, net_max = self().min().item(), self().max().item()
        a = 1.7159 / max(abs(net_min), abs(net_max))
        self.net = torch.nn.Sequential(self.net, ScaledTanh(a))

        
    def forward(self):
        return self.net(self.inputs)


def run_experiment(args, lr):
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    if args.function == "energy":
        the_function = energy(args.dimension)
    else:
        the_function = test_functions[args.function]

    network = EmbeddingPerceptron([args.n_embeddings] +
                                  [args.n_hiddens] * args.n_layers +
                                  [args.dimension])

    optimizer = torch.optim.SGD(network.parameters(), lr)

    scheduler = ReduceLROnPlateau(optimizer,
                                  patience=1,
                                  factor=0.99,
                                  threshold=1e-10,
                                  min_lr=1e-10,
                                  threshold_mode='abs')

    time_start = time.time()

    history = []

    for i in range(args.n_iters):
        errors = the_function(network())
        mean_error = errors.mean()
        min_error = errors.min()

        optimizer.zero_grad()
        mean_error.backward()
        optimizer.step()

        history.append([i, time.time() - time_start, min_error.item()])

        if min_error.item() != min_error.item():
            return None

        scheduler.step(mean_error.item())

    return torch.Tensor(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep convexity')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--function', type=str, default="ackley")
    parser.add_argument('--dimension', type=int, default=50)

    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--min_lr', type=float, default=-6)
    parser.add_argument('--max_lr', type=float, default=1)
    parser.add_argument('--n_lr', type=float, default=20)

    parser.add_argument('--n_embeddings', type=int, default=50)
    parser.add_argument('--n_hiddens', type=int, default=50)
    parser.add_argument('--n_layers', type=int, default=0)

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.set_default_tensor_type(torch.DoubleTensor)

    best_history = None
    best_lr = None 
    best_err = 1e10

    for lr in torch.logspace(args.min_lr, args.max_lr, args.n_lr):
        history = run_experiment(args, lr)

        if history is not None:
            err = history[:, -1].min().item()
            if err < best_err:
                best_lr = lr
                best_err = err
                best_history = history
            #print("+ Success at lr={:1.0e} with value {:.5f}".format(lr, err))
        #else:
            #print("- Failure at lr={:1.0e}".format(lr))
            
    print("* Best run at lr={:1.0e} with value {:.5f}".format(best_lr, best_err))

    
    fname = "f={}_emb={}_lay={}_hid={}_seed={}.txt".format(args.function,
                                                           args.n_embeddings,
                                                           args.n_layers,
                                                           args.n_hiddens,
                                                           args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, fname), "w") as f:
        f.write("# {}\n".format(str(vars(args))))
        for line in best_history:
            f.write("{:1.0e} {:.0f} {:.5f} {:.5f}\n".format(best_lr, *line))
