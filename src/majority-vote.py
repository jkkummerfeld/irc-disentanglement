#!/usr/bin/env python3

import sys

from collections import defaultdict

doing_graphs = True
counts = {}
all_nums = {}
file_count = 0
for line in sys.stdin:
    file_count += 1
    filename = line.strip()
    for line in open(filename):
        name, values = line.strip().split(":")
        if '-' in values:
            # Doing graphs:
            nums = values.split()
            info = counts.setdefault((name, nums[0]), {})
            if len(nums) > 2:
                for num in nums[2:]:
                    if num not in info:
                        info[num] = 0
                    info[num] += 1
            else:
                if nums[0] not in info:
                    info[nums[0]] = 0
                info[nums[0]] += 1
            seen = all_nums.setdefault(name, set())
            for num in nums:
                if num != '-':
                    seen.add(num)
        else:
            doing_graphs = False
            clusters = counts.setdefault(name, {})
            values = values.split()
            values.sort()
            cluster = tuple(values)
            if cluster not in clusters:
                clusters[cluster] = 0
            clusters[cluster] += 1
            seen = all_nums.setdefault(name, set())
            for num in cluster:
                if num != '-':
                    seen.add(num)

MIN_AGREE = int(sys.argv[1])
if doing_graphs:
    for name, src in counts:
        info = counts[name, src]
        options = [(count, num) for num, count in info.items()]
        options.sort(reverse=True)
        keep = []
        if options[0][0] >= MIN_AGREE:
            if options[0][1] != src:
                keep = [n for c, n in options if c >= MIN_AGREE and n != src]
        elif options[0][1] != src:
            keep = [options[0][1]]
        print("{}:{} - {}".format(name, src, ' '.join(keep)))
else:
    for name in counts:
        seen = all_nums[name]
        clusters = counts[name]
        ordered = []
        for cluster in clusters:
            count = clusters[cluster]
            if count >= MIN_AGREE:
                ordered.append((count, cluster))
        ordered.sort()
        # Go through from most agreement to least, and include all of the
        # cluster that hasn't been used yet
        included = set()
        for _, cluster in ordered:
            if all(n not in included for n in cluster):
                print("{}:{}".format(name, ' '.join(cluster)))
                for num in cluster:
                    included.add(num)
        for num in seen:
            if num not in included:
                print("{}:{}".format(name, num))
