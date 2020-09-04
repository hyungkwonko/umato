#source: https://github.com/ZJULearning/AtSNE/blob/master/tools/txt_to_fvecs.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert txt file to fvecs/ivecs file.
fvecs/ivecs vector file formats: http://corpus-texmex.irisa.fr/
"""
import argparse
import struct


def main():
    parser = argparse.ArgumentParser(description='Convert txt format file to fvecs')
    parser.add_argument('--data', type=str, help='data name')
    parser.add_argument('--ignore_firstline', action='store_true')
    parser.add_argument('--label', action='store_true')
    parser.add_argument('--dim', type=int, help='data dimension', default=0)
    args = parser.parse_args()

    dim = args.dim
    with open(f'{args.data}.txt', 'r') as input_file:
        with open(f'{args.data}.fvecs', 'wb') as output_file:
            if args.label:
                dim = 0
                output_file.write(struct.pack('=I', 0))  # placeholder

                for line in input_file.readlines():
                    if line:
                        dim += 1
                        output_file.write(struct.pack('=I', int(line)))
                output_file.seek(0)
                output_file.write(struct.pack('=I', dim))
            else:
                for index, line in enumerate(input_file.readlines()):
                    if index == 0:
                        sp = line.split()
                        if args.ignore_firstline:
                            if dim == 0 and len(sp) == 2:
                                dim = int(line.split()[1])
                            continue
                        else:
                            if dim == 0:
                                dim = len(sp)
                        if args.dim == 0:
                            print('guess data dimension {}'.format(dim))

                    line = line.split()
                    line = list(map(float, line[:dim]))
                    output_file.write(struct.pack('=I', dim))
                    output_file.write(struct.pack('={}{}'.format(dim, 'f'), *line))


if __name__ == '__main__':
    main()