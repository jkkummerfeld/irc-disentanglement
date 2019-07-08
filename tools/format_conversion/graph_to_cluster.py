#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    parser.add_argument("--no-cutoff", help="Do not enforce a cutoff based on source numbers", action="store_true")
    args = parser.parse_args()

    filename = ""
    nodes = {}
    edges = {}
    cutoffs = {}
    for line in sys.stdin:
        if ':' in line:
            filename, line = line.split(':')

        parts = [int(v) for v in line.strip().split() if v != '-']
        source = max(parts)
        nodes.setdefault(filename, set()).add(source)
        parts.remove(source)
        for num in parts:
            edges.setdefault(filename, []).append((source, num))
            nodes.setdefault(filename, set()).add(num)

        # A cutoff based on the min source is used to get consistent sets for evaluation
        if filename not in cutoffs:
            cutoffs[filename] = source
        else:
            cutoffs[filename] = min(source, cutoffs[filename])

    for filename in nodes:
        cutoff = cutoffs[filename]
        clusters = union_find(nodes[filename], edges[filename])
        for cluster in clusters:
            vals = [str(v) for v in cluster if v >= cutoff or args.no_cutoff]
            vals.sort()
            print(filename +":"+ " ".join(vals))

