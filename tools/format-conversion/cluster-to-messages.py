#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a conversation graph to a set of connected components (i.e. threads).')
    parser.add_argument('raw_data', help='File containing the raw log content as <filename>:...')
    args = parser.parse_args()

    clusters = {}
    for line in sys.stdin:
        filename = ''
        if ':' in line:
            filename, line = line.split(":")
        nums = [int(v) for v in line.strip().split()]
        nums.sort()
        clusters.setdefault(filename, []).append(nums)

    text = {}
    for line in open(args.raw_data):
        filename = ''
        if ':' in line and line[0] not in '[=':
            parts = line.split(":")
            filename = parts[0]
            line = ':'.join(parts[1:])
        text.setdefault(filename, []).append(line.strip())

    for filename in clusters:
        sortable_clusters = []
        for cluster in clusters[filename]:
            size = len(cluster)
            first = min(cluster)
            sortable_clusters.append((first, size, cluster))
        sortable_clusters.sort()

        for _, _, cluster in sortable_clusters:
            print(cluster)
            for num in cluster:
                print(text[filename][num])
            print()
