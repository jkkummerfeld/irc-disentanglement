#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys

def add_pairs(filename, all_pairs):
    basefile, annotator = filename.split('.annotation.')
    basefile = basefile.split('/')[-1]
    pairs = all_pairs.setdefault(annotator, {}).setdefault(basefile, set())

    for line in open(filename):
        parts = [int(v) for v in line.strip().split() if v != '-']
        source = max(parts)
        parts.remove(source)
        for part in parts:
            pairs.add((part, source))
    return all_pairs

def print_kappa(name, annotations0, annotations1):
    # Cohen's Kappa
    a, b, c, d = 0, 0, 0, 0
    # These will be:
    #                      Rater 0
    #                   link   no-link
    # Rater 1   link      a      c
    #          no-link    b      d

    # Get valid ranges for annotations
    filenames = list(annotations0.keys()) + list(annotations1.keys())
    for filename in filenames:
        min_src = 10000000
        max_src = 0
        for anno in [annotations0, annotations1]:
            if filename in anno:
                for _, src in anno[filename]:
                    min_src = min(min_src, src)
                    max_src = max(max_src, src)
            else:
                print("Comparing annotations without matching files", name, file=sys.stderr)
        for i in range(min_src, max_src + 1):
            for j in range(0, i + 1):
                if (j, i) in annotations0[filename]:
                    if (j, i) in annotations1[filename]: a += 1
                    else: b += 1
                else:
                    if (j, i) in annotations1[filename]: c += 1
                    else: d += 1

    # Kappa
    total = a + b + c + d
    a /= total
    b /= total
    c /= total
    d /= total
    pO = (a + d)
    pE = (a + b) * (a + c) + (c + d) * (b + d)
    kappa = (pO - pE) / (1 - pE)
    print("{} Kappa {:.3f}   from {} {} {} {}".format(name, kappa, a, b, c, d))

def print_f1(name, annotations0, annotations1):
    # Harmonic mean of the two values for how many we had in common
    common, total_0, total_1 = 0, 0, 0

    # Get valid ranges for annotations
    filenames = list(annotations0.keys()) + list(annotations1.keys())
    for filename in filenames:
        if filename in annotations0 and filename in annotations1:
            common += len(annotations0[filename].intersection(annotations1[filename]))
        if filename in annotations0:
            total_0 += len(annotations0[filename])
        if filename in annotations1:
            total_1 += len(annotations1[filename])
    val0 = 100 * common / total_0
    val1 = 100 * common / total_1
    f1 = 2 * val0 * val1 / (val0 + val1)

    print("{} F1    {:.2f}   from {:.2f} {:.2f} - {} {} {}".format(name, f1, val0, val1, common, len(annotations0), len(annotations1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare annotations of conversation graphs')
    parser.add_argument('files', help='Files containing annotations', nargs="+")
    parser.add_argument('-f1', help='Calculate f1', action='store_true')
    args = parser.parse_args()

    all_pairs = {}
    for filename in args.files:
        add_pairs(filename, all_pairs)

    annotators = list(all_pairs.keys())
    for annotator0 in annotators:
        for annotator1 in annotators:
            if annotator0 == annotator1:
                break
            name = "Annotators:"+ annotator0 +":"+ annotator1
            annotations0 = all_pairs[annotator0]
            annotations1 = all_pairs[annotator1]
            print_kappa(name, annotations0, annotations1)
            if args.f1:
                print_f1(name, annotations0, annotations1)

