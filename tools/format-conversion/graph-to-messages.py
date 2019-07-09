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
    parser = argparse.ArgumentParser(description='Go from graphannotations to conversations')
    parser.add_argument('ascii', help="File containing the ascii version of the data")
    parser.add_argument('annotations', help="File containing the graph annotations")
    parser.add_argument('--keep-sys', help="Keep conversations that are just system messages", action='store_true')
    parser.add_argument('--max-speakers', help="Only print conversations with this many or fewer participants (not counting the channel bot)", type=int)
    parser.add_argument('--min-speakers', help="Only print conversations with this many or more participants (not counting the channel bot)", type=int)
    args = parser.parse_args()

    # Hold-over from multi-file code
    filename = ""

    nodes = {}
    edges = {}
    cutoffs = {}
    for line in open(args.annotations):
        parts = [int(v) for v in line.strip().split() if v != '-']
        source = max(parts)
        nodes.setdefault(filename, set()).add(source)
        parts.remove(source)
        for num in parts:
            edges.setdefault(filename, []).append((source, num))
            nodes.setdefault(filename, set()).add(num)

    text = {}
    for line in open(args.ascii):
        text.setdefault(filename, []).append(line.strip())

    for filename in nodes:
        clusters = union_find(nodes[filename], edges[filename])
        for cluster in clusters:
            if args.max_speakers or args.min_speakers:
                speakers = set()
                for num in cluster:
                    line = text[filename][num]
                    speaker = line.split()[1]
                    if line.startswith("["):
                        speaker = speaker[1:-1]
                    if speaker not in {'ubotu', 'ubottu'}:
                        speakers.add(speaker)
                if args.max_speakers and len(speakers) > args.max_speakers:
                    continue
                if args.min_speakers and len(speakers) < args.min_speakers:
                    continue
            if not args.keep_sys:
                all_sys = True
                for num in cluster:
                    if text[filename][num].startswith("["):
                        all_sys = False
                if all_sys:
                    continue

            for num in cluster:
                print(text[filename][num])
            print()

