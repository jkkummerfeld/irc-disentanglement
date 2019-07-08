#!/usr/bin/env python3

from __future__ import print_function

import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a cluster to a graph by linking each message to the one before it.')
    args = parser.parse_args()

    for line in sys.stdin:
        filename, nums = line.split(":")
        nums = [int(n) for n in nums.split()]
        nums.sort()
        print("{}:{} - {}".format(filename, nums[0], ""))
        for n0, n1 in zip(nums[1:], nums[:-1]):
            print("{}:{} - {}".format(filename, n0, n1))
