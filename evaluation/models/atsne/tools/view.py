#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw point in 2D space."""
import argparse
import os
import os.path
import concurrent.futures
import multiprocessing
import random

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # We can use this tool without GUI. Do not change order of these lines
import matplotlib.pyplot as plt


def read_file(srcpath, ignore_first_line=False, focus=None):
    xs = []
    ys = []
    with open(srcpath, 'r') as f:
        for linenum, line in enumerate(f.readlines()):
            if ignore_first_line:
                linenum -= 1
                if linenum < 0:
                    continue
            line = line.split()
            x, y = line[:2]
            if focus is None or linenum in focus:
                xs.append(float(x))
                ys.append(float(y))
    return xs, ys


def draw_picture(srcpath, imgpath, title, colors=None, ignore_first_line=False, colormap='rainbow', size=1, focus=None,
                 draw_grad=False, next_src_path=None, figsize=24, noaxis=False):
    print(srcpath)
    xs, ys = read_file(srcpath, ignore_first_line, focus)
    if next_src_path:
        dxs, dys = read_file(next_src_path, ignore_first_line, focus)
        for i in range(len(xs)):
            dxs[i] -= xs[i]
            dys[i] -= ys[i]
    grad_pos_x = []
    grad_pos_y = []
    grad_neg_x = []
    grad_neg_y = []
    if draw_grad:
        with open(srcpath + '.grad', 'r') as f:
            for linenum, line in enumerate(f.readlines()):
                line = line.split()
                x1, y1, x2, y2 = line[:4]
                if focus is None or linenum in focus:
                    grad_pos_x.append(-float(x1))
                    grad_pos_y.append(-float(y1))
                    grad_neg_x.append(-float(x2))
                    grad_neg_y.append(-float(y2))
    plt.figure(figsize=(figsize, figsize))
    plt.gca().set_aspect('equal', adjustable='box')
    if noaxis:
        plt.axis('off')
    plt.title(title, size=30)
    plt.xlabel('X', size=20)
    plt.ylabel('Y', size=20)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    kwargs = dict()

    if next_src_path:
        plt.quiver(xs, ys, dxs, dys, units='dots', width=size, angles='xy', scale_units='xy', scale=1, color='g')
    if draw_grad:
        plt.quiver(xs + xs, ys + ys, grad_pos_x + grad_neg_x, grad_pos_y + grad_neg_y, color=['r'] * len(xs) + ['b'] * len(xs),
                   # scale_units='xy', scale=1,
                   units='dots', width=size)
        # plt.quiver(xs, ys, grad_pos_x, grad_pos_y, color='r', units='dots', width=size)
        # plt.quiver(xs, ys, grad_neg_x, grad_neg_y, color='b', units='dots', width=size)
    if colors:
        # candidate color maps: https://matplotlib.org/users/colormaps.html
        # Suggestion: rainbow, cubehelix, gist_rainbow, hsv, gist_ncar
        kwargs.update(dict(c=colors, cmap=plt.get_cmap(colormap)))

    if not next_src_path:
        plt.scatter(xs, ys, s=size, **kwargs)

    plt.savefig(imgpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Drap point in 2D space')
    parser.add_argument('datafile', type=str, help='filepath prefix of data file')
    parser.add_argument('--label', type=str, help='location label file')
    parser.add_argument('--colormap', type=str, default='rainbow', help='colormap')
    parser.add_argument('--ignore_first_line', action='store_true', help='ignore the first line in file (for largevis format)')
    parser.add_argument('--focus', type=str, default=None, help='show some point only, point id split by comma')
    parser.add_argument('--size', type=float, default=1.0, help='point size')
    parser.add_argument('--figsize', type=float, default=24.0, help='figure size')
    parser.add_argument('--random_color', action='store_true')
    parser.add_argument('--draw_grad', action='store_true', help='Draw gradient arrrow')
    parser.add_argument('--arrow', action='store_true', help='Draw arrow to next figure')
    parser.add_argument('--parallel', type=int, default=multiprocessing.cpu_count(), help='parallel worker to draw figure')
    parser.add_argument('--noaxis', action='store_true', help='hide axis')
    parser.add_argument('--title', type=str, help='title in image', default='')
    args = parser.parse_args()

    focus = args.focus
    if focus:
        focus = list(map(int, focus.split(',')))
        print('focus on', focus)

    labels = None
    if args.label:
        labels = []
        with open(args.label, 'r') as f:
            for linenum, line in enumerate(f.readlines()):
                if line and (focus is None or linenum in focus):
                    labels.append(int(line))

    if args.random_color and focus:
        labels = list(range(len(focus)))
        random.shuffle((labels))

    dirname = os.path.dirname(args.datafile)
    basename = os.path.basename(args.datafile)
    if dirname == "":  # datafile in current dir
        dirname = '.'
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.parallel) as executor:
        filenames = sorted(os.listdir(dirname))
        filenames = filter(lambda x: x.startswith(basename) and not x.endswith('.png'), filenames)
        filenames = filter(lambda x: len(x[len(basename):]) == 0 or x[len(basename) + 1:].isdigit(), filenames)
        filenames = list(filenames)
        for index, filename in enumerate(filenames):
            imgpath = os.path.join(dirname, filename + ".png")
            if os.path.exists(imgpath):
                continue
            # print(os.path.join(dirname, filename))
            next_src_path = filenames[index + 1] if args.arrow else None
            title = filename
            if args.title:
                title = title.replace(basename, args.title)
            kwargs = dict(
                srcpath=os.path.join(dirname, filename),
                imgpath=imgpath,
                title=title,
                colors=labels,
                ignore_first_line=args.ignore_first_line,
                colormap=args.colormap,
                size=args.size,
                focus=focus,
                draw_grad=args.draw_grad,
                next_src_path=next_src_path,
                figsize=args.figsize,
                noaxis=args.noaxis,
            )
            if args.parallel > 1:
                futures.append(executor.submit(draw_picture, **kwargs))
            else:
                draw_picture(**kwargs)
    for f in futures:
        f.result()


if __name__ == '__main__':

    main()
