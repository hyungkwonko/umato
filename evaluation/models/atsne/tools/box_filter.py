#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get labels of data points in a specific bounding box"""
import argparse


def main():
    parser = argparse.ArgumentParser(description='box filter tool')
    parser.add_argument('datafile', type=str, help='filepath prefix of data file')
    parser.add_argument('x1', type=float)
    parser.add_argument('x2', type=float)
    parser.add_argument('y1', type=float)
    parser.add_argument('y2', type=float)
    parser.add_argument('--label', type=str, help='location label file')
    args = parser.parse_args()

    x1 = min(args.x1, args.x2)
    x2 = max(args.x1, args.x2)
    y1 = min(args.y1, args.y2)
    y2 = max(args.y1, args.y2)

    with open(args.datafile, 'r') as f:
        for linenumber, line in enumerate(f.readlines()):
            line = line.split()
            x, y = line[:2]
            x = float(x)
            y = float(y)
            if x1 <= x <= x2 and y1 <= y <= y2:
                print(x, y, linenumber, line[-1])


if __name__ == '__main__':

    main()
