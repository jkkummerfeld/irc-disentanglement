#!/usr/bin/env python3

from __future__ import print_function

import argparse
import random
import sys

def read_file(name):
    links = {}
    for line in open(name):
        filename, info = line.strip().split(':')
        filename = filename.split("/")[-1]
        targets = [int(n) for n in info.split() if n != '-']
        src = max(targets)
        targets.remove(src)
        to_save = links.setdefault((filename, src), [])
        for target in targets:
            to_save.append(target)
    for key, targets in links.items():
        if len(targets) == 0:
            targets.append(key[1])
    return links

def remix(keys, a, b):
    out0 = {}
    out1 = {}
    for key in keys:
        a_val = None
        if len(a) == 0:
            a_val = a[0][key]
        else:
            a_val = a[random.randint(0, len(a) - 1)][key]
        b_val = None
        if len(b) == 0:
            b_val = b[0][key]
        else:
            b_val = b[random.randint(0, len(b) - 1)][key]
        if random.random() > 0.5:
            out0[key] = a_val
            out1[key] = b_val
        else:
            out0[key] = b_val
            out1[key] = a_val
    return out0, out1

def score(gold, autos):
    total_p = 0
    total_r = 0
    total_f = 0
    for auto in autos:
        total_gold = 0
        total_auto = 0
        matched = 0

        for key, targets in gold.items():
            for target in targets:
                if target in auto[key]:
                    matched += 1
            total_gold += len(targets)
        for key, targets in auto.items():
            total_auto += len(targets)

        p = 0.0
        if total_auto > 0:
            p = 100 * matched / total_auto
        r = 0.0
        if total_gold > 0:
            r = 100 * matched / total_gold
        f = 0.0
        if matched > 0:
            f = 2 * p * r / (p + r)
        total_p += p
        total_r += r
        total_f += f
    return total_p / len(autos), total_r / len(autos), total_f / len(autos)

def sub(scores0, scores1):
    combined = []
    for a, b, in zip(scores0, scores1):
        combined.append(abs(a - b))
    return combined

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate conversation graph output.')
    parser.add_argument('gold', help='Files containing annotations')
    parser.add_argument('--auto0', help='Files containing system output', nargs="+")
    parser.add_argument('--auto1', help='Files containing system output', nargs="+")
    args = parser.parse_args()

    gold = read_file(args.gold)
    data0 = [read_file(name) for name in args.auto0]
    data1 = [read_file(name) for name in args.auto1]

    data0_score = score(gold, data0)
    data1_score = score(gold, data1)

    print("Data 0 scores:", data0_score)
    print("Data 1 scores:", data1_score)
    base = sub(data0_score, data1_score)
    print("Original difference:", base)

    counts = [0, 0, 0]
    tests = 1000
    keys = [key for key in gold]
    for i in range(tests):
        pair = remix(keys, data0, data1)
        diffs = None
        diffs = sub(score(gold, [pair[0]]), score(gold, [pair[1]]))
        for i in range(3):
            if diffs[i] >= base[i]:
                counts[i] += 1

    for i in range(3):
        print(counts[i] / tests, end=' ')
    print()
