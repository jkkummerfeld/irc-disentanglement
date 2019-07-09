#!/usr/bin/env python3

from __future__ import print_function

import argparse
import logging
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the output from a run of the system into content that can be evaluated.')
    args = parser.parse_args()

    done_training = False
    for line in sys.stdin:
        if line.strip() == "Finished training.":
            done_training = True
        elif done_training and '[' in line and ']' in line:
            line = line.split("[")[-1].split('/')[-1].split("]")[0].strip()
            parts = line.split()
            print("{}:{}".format(parts[0], ' '.join(parts[1:])))
