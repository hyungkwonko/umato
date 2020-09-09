#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Coil20 dataset to .fvecs file."""
import sys
import os.path
import re
import struct

import numpy as np
import PIL.Image


def get_label_by_name(name):
    g = re.match('obj(\d+)__\d+.png', name)
    return int(g.group(1))


def main():
    image_dir = sys.argv[1]
    assert(os.path.isdir(image_dir))
    labels = []
    with open(os.path.join(image_dir, 'coil20.fvecs'), 'wb') as f:
        for filename in sorted(os.listdir(image_dir)):
            if not filename.endswith('png'):
                continue
            label = get_label_by_name(filename)
            labels.append(label)

            image = PIL.Image.open(os.path.join(image_dir, filename))
            image = np.array(np.array(image) / 255, dtype='float32').flatten()
            f.write(struct.pack('=I', image.size))
            f.write(struct.pack('={}f'.format(image.size), *image))

            print(filename)

    with open(os.path.join(image_dir, 'coil20.label'), 'wb') as f:
        f.write(struct.pack('=I', len(labels)))
        labels = np.array(labels, dtype='int32')
        f.write(struct.pack('={}I'.format(labels.size), *labels))

    print('{} image transformed.'.format(labels.size))


if __name__ == '__main__':
    main()
