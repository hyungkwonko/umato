#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert txt vector file to fvecs/ivecs.
fvecs/ivecs vector file formats: http://corpus-texmex.irisa.fr/
"""
import sys
import struct
import argparse


def conver_fvec_to_largevis(input_file, output_file, N=None, f='f'):
    dim = struct.unpack("<I", input_file.read(4))[0]
    if not N:
        input_file.seek(0, 2)
        file_size = input_file.tell()
        N = int(file_size / (1 + dim) / 4)
        input_file.seek(4, 0)
    print("N = ", N, " dim = ", dim)
    output_file.write("{} {}\n".format(N, dim))

    iterline = range(N)
    try:
        from tqdm import tqdm
    except Exception as e:
        print("install tqdm by:\n pip install tqdm")
    else:
        iterline = tqdm(iterline)

    for i in iterline:
        if i != 0:
            input_file.read(4)
        line = input_file.read(dim * 4)
        line = struct.unpack("<{}{}".format(dim, f), line)
        output_file.write(" ".join(map(str, line)))
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Preprocess dataset for largevis from fvecs')
    parser.add_argument('input_file', type=str, help='location of source dataset')
    parser.add_argument('output_file', type=str, help='location of target dataset')
    parser.add_argument('--label', action='store_true')
    args = parser.parse_args()

    with open(args.input_file, 'rb') as input_file:
        with open(args.output_file, 'w') as output_file:
            f = 'I' if args.label else 'f'
            conver_fvec_to_largevis(input_file, output_file, f=f)


if __name__ == '__main__':
    main()
