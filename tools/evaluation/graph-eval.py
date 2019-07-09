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

        nums = [int(n) for n in line.split() if n != '-']
        source = max(nums)
        nums.remove(source)
        if len(nums) == 0 or (nums[0] == source and len(nums) == 1):
            storage.setdefault(filename, set()).add((source, source))
        else:
            for num in nums:
                storage.setdefault(filename, set()).add((source, num))

def read_raw(filename, storage, gold):
    raw_messages = []
    users = set()
    for line in open(filename):
        line = line.strip()
        tokens = line.split()
        assert len(tokens) > 0, "Blank line in text file {}".format(filename)
        time = (None, None)
        user = tokens[1]
        if tokens[0] != '===':
            hour = int(tokens[0][1:-4])
            minute = int(tokens[0][4:-1])
            time = (hour, minute)
            user = user[1:-1]
        if len(user) == 0:
            user = tokens[2]
        raw_messages.append((time, user, line))
        users.add(user)

    directed = set()
    filename = filename.split('/')[-1]
    links = gold[filename]
    for source, target in links:
        if source == target:
            continue
        message = raw_messages[source][2]
        target_user = raw_messages[target][1]
        if target_user in message:
            directed.add(source)

    storage[filename] = directed

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
    parser.add_argument('--gold', help='Files containing annotations', nargs="+")
    parser.add_argument('--auto', help='Files containing system output', nargs="+")
    parser.add_argument('--raw', help='Files containing raw data, for additional statistics', nargs="*")
    args = parser.parse_args()

    assert args.gold is not None and args.auto is not None

    gold = {}
    for filename in args.gold:
        get_pairs(filename, gold)

    auto = {}
    for filename in args.auto:
        get_pairs(filename, auto)

    raw = {}
    if args.raw is not None:
        for filename in args.raw:
            read_raw(filename, raw, gold)
    else:
        for filename in gold:
            raw[filename] = set()

###    for filename in auto:
###        for pair in auto[filename]:
###            print(filename, pair)
###            min(pair), '-', max(pair))

    total_gold = 0
    total_auto = 0
    matched = 0

    total_gold_directed = 0
    total_auto_directed = 0
    matched_directed = 0

    total_gold_undirected = 0
    total_auto_undirected = 0
    matched_undirected = 0

    for filename in gold:
        if filename in auto:
            for pair in gold[filename]:
                if pair[0] in raw[filename]:
                    total_gold_directed += 1
                else:
                    total_gold_undirected += 1
                if pair in auto[filename]:
                    matched += 1
                    if pair[0] in raw[filename]:
                        matched_directed += 1
                    else:
                        matched_undirected += 1
        total_gold += len(gold[filename])
    for filename in auto:
        total_auto += len(auto[filename])
        for pair in auto[filename]:
            if pair[0] in raw[filename]:
                total_auto_directed += 1
            else:
                total_auto_undirected += 1

    print_results("", total_gold, total_auto, matched)
    print_results("_directed", total_gold_directed, total_auto_directed, matched_directed)
    print_results("_undirected", total_gold_undirected, total_auto_undirected, matched_undirected)

###    print("TODO:")
###    print("Show performance breakdown depending on number of antecedents in gold (0, 1, 2+)")
