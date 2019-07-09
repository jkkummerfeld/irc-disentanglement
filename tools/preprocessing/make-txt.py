#!/usr/bin/env python3

from __future__ import print_function

import argparse
import sys
import string
import unicodedata

def convert_word(word):
    nword = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode("ascii").strip()
    nword = ''.join(nword.split())
    if len(nword) == 0:
        nword = "<unconvertable>"
    elif word.startswith(nword) and len(nword) != len(word):
        nword += " <unconvertable>"
    elif word.endswith(nword) and len(nword) != len(word):
        nword = "<unconvertable> " + nword
    return nword

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Remove non-ascii content from IRC logs. Also do: tr -cd '[:print:]\\n\\t' ")
    parser.add_argument('--output_suffix', help='Save to files with the suffix added.')
    parser.add_argument('raw_data', help='File containing the raw logs.', nargs="+")
    args = parser.parse_args()

    out = sys.stdout
    for input_filename in args.raw_data:
        if args.output_suffix is not None:
            out = open(input_filename + args.output_suffix, 'w')
        data = open(input_filename)
        line = ''
        while line is not None:
            try:
                line = data.readline()
                if line == '':
                    line = None
                    break
                edited = []
                cur = []
                for char in line:
                    if char in ' \t\n':
                        if len(cur) > 0:
                            edited.append(convert_word(''.join(cur)))
                        edited.append(char)
                        cur = []
                    else:
                        cur.append(char)
                to_print = ''.join(edited)
                if to_print[-1] == '\n':
                    print(to_print, file=out, end='')
                else:
                    print(to_print, file=out)
            except UnicodeDecodeError:
                pass
            except:
                pass
        if args.output_suffix is not None:
            out.close()

