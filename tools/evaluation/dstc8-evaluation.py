#!/usr/bin/env python3

# For the definition of the metric, see https://en.wikipedia.org/wiki/Variation_of_information

from __future__ import print_function

import argparse
import math
import sys

def find(x, parents):
    while parents[x] != x:
        parent = parents[x]
        parents[x] = parents[parent]
        x = parent
    return x

def union(x, y, parents, sizes):
    # Get the representative for their sets
    x_root = find(x, parents)
    y_root = find(y, parents)
 
    # If equal, no change needed
    if x_root == y_root:
        return
 
    # Otherwise, merge them
    if sizes[x_root] > sizes[y_root]:
        parents[y_root] = x_root
        sizes[x_root] += sizes[y_root]
    else:
        parents[x_root] = y_root
        sizes[y_root] += sizes[x_root]

def union_find(nodes, edges):
    # Make sets
    parents = {n:n for n in nodes}
    sizes = {n:1 for n in nodes}

    for edge in edges:
        union(edge[0], edge[1], parents, sizes)

    clusters = {}
    for n in parents:
        clusters.setdefault(find(n, parents), set()).add(n)
    cluster_list = list(clusters.values())
    return cluster_list

def read_data(filenames):
    clusters = {}
    graphs = {}
    all_points = set()
    for filename in filenames:
        nodes = {}
        edges = {}
        for line in open(filename):
            if line.startswith("#") or line.startswith("%") or line.startswith("/"):
                continue
            cfile = filename
            if ':' in line:
                cfile, line = line.split(':')
            cfile = cfile.split('/')[-1]
            parts = [int(v) for v in line.strip().split() if v != '-']
            assert len(parts) == 2
            source = max(parts)
            nodes.setdefault(cfile, set()).add(source)
            parts.remove(source)
            for num in parts:
                edges.setdefault(cfile, []).append((source, num))
                nodes.setdefault(cfile, set()).add(num)
                graphs.setdefault(cfile, {}).setdefault(source, set()).add(num)

        for cfile in nodes:
            for cluster in union_find(nodes[cfile], edges[cfile]):
                vals = {v for v in cluster if v >= 1000}
                clusters.setdefault(cfile, []).append(vals)
                for val in vals:
                    all_points.add("{}:{}".format(cfile, val))
    return clusters, all_points, graphs

def clusters_to_contingency(gold, auto):
    # A table, in the form of:
    # https://en.wikipedia.org/wiki/Rand_index#The_contingency_table
    table = {}
    for filename in auto:
        for i, acluster in enumerate(auto[filename]):
            aname = "auto.{}.{}".format(filename, i)
            current = {}
            table[aname] = current
            for j, gcluster in enumerate(gold[filename]):
                gname = "gold.{}.{}".format(filename, j)
                count = len(acluster.intersection(gcluster))
                if count > 0:
                    current[gname] = count
    counts_a = {}
    for filename in auto:
        for i, acluster in enumerate(auto[filename]):
            aname = "auto.{}.{}".format(filename, i)
            counts_a[aname] = len(acluster)
    counts_g = {}
    for filename in gold:
        for i, gcluster in enumerate(gold[filename]):
            gname = "gold.{}.{}".format(filename, i)
            counts_g[gname] = len(gcluster)
    return table, counts_a, counts_g

def variation_of_information(contingency, row_sums, col_sums):
    total = 0.0
    for row in row_sums:
        total += row_sums[row]

    H_UV = 0.0
    I_UV = 0.0
    for row in contingency:
        for col in contingency[row]:
            num = contingency[row][col]
            H_UV -= (num / total) * math.log(num / total, 2)
            I_UV += (num / total) * math.log(num * total / (row_sums[row] * col_sums[col]), 2)

    H_U = 0.0
    for row in row_sums:
        num = row_sums[row]
        H_U -= (num / total) * math.log(num / total, 2)
    H_V = 0.0
    for col in col_sums:
        num = col_sums[col]
        H_V -= (num / total) * math.log(num / total, 2)

    max_score = math.log(total, 2)
    VI = H_UV - I_UV

    scaled_VI = VI / max_score
    print("{:5.2f}   1 - Scaled VI".format(100 - 100 * scaled_VI))

def adjusted_rand_index(contingency, row_sums, col_sums):
    # See https://en.wikipedia.org/wiki/Rand_index
    rand_index = 0.0
    total = 0.0
    for row in contingency:
        for col in contingency[row]:
            n = contingency[row][col]
            rand_index += n * (n-1) / 2.0
            total += n
    
    sum_row_choose2 = 0.0
    for row in row_sums:
        n = row_sums[row]
        sum_row_choose2 += n * (n-1) / 2.0
    sum_col_choose2 = 0.0
    for col in col_sums:
        n = col_sums[col]
        sum_col_choose2 += n * (n-1) / 2.0
    random_index = sum_row_choose2 * sum_col_choose2 * 2.0 / (total * (total - 1))

    max_index = 0.5 * (sum_row_choose2 + sum_col_choose2)

    adjusted_rand_index = (rand_index - random_index) / (max_index - random_index)

    print('{:5.2f}   Adjusted rand index'.format(100 * adjusted_rand_index))

def exact_match(gold, auto):
    # P/R/F over complete clusters
    total_gold = 0
    total_matched = 0
    for filename in gold:
        for cluster in gold[filename]:
            if len(cluster) == 1:
                continue
            total_gold += 1
            matched = False
            for ocluster in auto[filename]:
                if len(ocluster.symmetric_difference(cluster)) == 0:
                    matched = True
                    break
            if matched:
                total_matched += 1
    total_auto = 0
    for filename in auto:
        for cluster in auto[filename]:
            if len(cluster) == 1:
                continue
            total_auto += 1
    p, r, f = 0.0, 0.0, 0.0
    if total_auto > 0:
        p = 100 * total_matched / total_auto
    if total_gold > 0:
        r = 100 * total_matched / total_gold
    if total_matched > 0:
        f = 2 * p * r / (p + r)
    print("{:5.2f}   Matched clusters precision".format(p))
    print("{:5.2f}   Matched clusters recall".format(r))
    print("{:5.2f}   Matched clusters f-score".format(f))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate cluster / thread / conversation metrics.')
    parser.add_argument('--gold', help='File(s) containing the gold clusters, one per line. If a line contains a ":" the start is considered a filename', required=True, nargs="+")
    parser.add_argument('--auto', help='File(s) containing the system clusters, one per line. If a line contains a ":" the start is considered a filename', required=True, nargs="+")
    parser.add_argument('--metric', nargs="+", choices=['vi', 'rand', 'ex', 'all'], default=['all'])
    args = parser.parse_args()

    gold, gpoints, gedges = read_data(args.gold)
    auto, apoints, aedges = read_data(args.auto)
    issue = False
    for filename in auto:
        if filename not in gold:
            print("Gold is missing file {}".format(filename), file=sys.stderr)
            issue = True
    for filename in gold:
        if filename not in auto:
            print("Auto is missing file {}".format(filename), file=sys.stderr)
            issue = True
    if issue:
        sys.exit(0)
    if len(apoints.symmetric_difference(gpoints)) != 0:
        print(apoints.difference(gpoints))
        print(gpoints.difference(apoints))
        raise Exception("Set of lines does not match: {}".format(apoints.symmetric_difference(gpoints)))

    contingency, row_sums, col_sums = None, None, None
    if 'vi' in args.metric or 'rand' in args.metric or 'all' in args.metric:
        contingency, row_sums, col_sums = clusters_to_contingency(gold, auto)

    if 'vi' in args.metric or 'all' in args.metric:
        variation_of_information(contingency, row_sums, col_sums)
    if 'rand' in args.metric or 'all' in args.metric:
        adjusted_rand_index(contingency, row_sums, col_sums)
    if 'ex' in args.metric or 'all' in args.metric:
        exact_match(gold, auto)

