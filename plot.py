# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
plt.rc('font', family='serif')
plt.rc('font', size=12)

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep convexity')
    parser.add_argument('--save_dir', type=str, default='results/')
    parser.add_argument('--plot_dir', type=str, default='plots/')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_hiddens', type=int, default=50)
    parser.add_argument('--n_embeddings', type=int, default=50)
    args = parser.parse_args()

    functions = [
        "ackley",
        "michal",
        "schwefel",
        "sphere",
        "tang",
        "rastrigin",
        "rosenbrock",
        "levy",
        "energy"
    ]

    plt.figure(figsize=(10, 6))

    for f, function in enumerate(functions):
        plt.subplot(3, 3, f + 1)

        shallows = glob.glob(os.path.join(args.save_dir,
                                          "f={}_emb={}_lay=0_hid={}_*".format(function,
                                                                       args.n_embeddings,
                                                                       args.n_hiddens)))

        deeps = glob.glob(os.path.join(args.save_dir,
                                       "f={}_emb={}_lay={}_hid={}_*".format(function,
                                                                     args.n_embeddings,
                                                                     args.n_layers,
                                                                     args.n_hiddens)))
        print(len(shallows), len(deeps))

        for shallow in shallows:
            data = np.genfromtxt(shallow)
            if len(data):
                data = data[:, 2:]
                plt.plot(data[:, 0], data[:, 1], color="C2", alpha=0.5)

        for deep in deeps:
            data = np.genfromtxt(deep)
            if len(data):
                data = data[:, 2:]
                plt.plot(data[:, 0], data[:, 1], color="C1", alpha=0.5)

        plt.xscale("log")
        if function == "rastrigin" or function == "rosenbrock":
            plt.yscale("log")

        plt.margins(0.05)
        plt.title(function)

    plt.tight_layout(pad=0)
    pdf_name = os.path.join(args.plot_dir,
                            'plot_{}_{}.pdf'.format(args.n_layers,
                                                    args.n_hiddens))

    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)

    plt.savefig(pdf_name)
