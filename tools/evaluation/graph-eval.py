#!/usr/bin/env python3

from __future__ import print_function

import argparse
import sys

def get_pairs(filename, storage):
    for line in open(filename):
        line = line.strip()
        if ':' in line:
            filename = line.split(':')[0]
            line = line.split(":")[1]
        elif line[0] == '#':
            continue

        nums = [int(n) for n in line.split() if n != '-']
        source = max(nums)
        nums.remove(source)
        if len(nums) == 0 or (nums[0] == source and len(nums) == 1):
            storage.setdefault(filename, set()).add((source, source))
        else:
            for num in nums:
                storage.setdefault(filename, set()).add((source, num))

def print_results(name, total_gold, total_auto, matched):
    p = 0.0
    if total_auto > 0:
        p = 100 * matched / total_auto
    r = 0.0
    if total_gold > 0:
        r = 100 * matched / total_gold
    f = 0.0
    if matched > 0:
        f = 2 * p * r / (p + r)
    print("g/a/m{}:".format(name), total_gold, total_auto, matched)
    print("p/r/f{}: {:.3} {:.3} {:.3}".format(name, p, r, f))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate conversation graph output.')
    parser.add_argument('--gold', help='Files containing annotations', nargs="+", required=True)
    parser.add_argument('--auto', help='Files containing system output', nargs="+", required=True)
    args = parser.parse_args()

    assert args.gold is not None and args.auto is not None

    # Read
    gold = {}
    for filename in args.gold:
        get_pairs(filename, gold)

    auto = {}
    for filename in args.auto:
        get_pairs(filename, auto)

    # Calculate
    total_gold = 0
    for filename in gold:
        total_gold += len(gold[filename])

    total_auto = 0
    for filename in auto:
        total_auto += len(auto[filename])

    matched = 0
    for filename in gold:
        if filename in auto:
            for pair in gold[filename]:
                if pair in auto[filename]:
                    matched += 1

    print_results("", total_gold, total_auto, matched)
