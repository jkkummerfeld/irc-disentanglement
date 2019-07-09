#!/usr/bin/env python3

# For the definition of the metric, see https://en.wikipedia.org/wiki/Variation_of_information

from __future__ import print_function

import argparse
import math
import sys

from ortools.graph import pywrapgraph
from sklearn import metrics

def read_clusters(filename):
    # Read provided data
    clusters = {}
    cfile = ""
    all_points = set()
    for line in open(filename):
        if ':' in line:
            cfile = ':'.join(line.split(':')[:-1]).split('/')[-1]
            line = line.split(":")[-1]
        cluster = {int(v) for v in line.split()}
        clusters.setdefault(cfile, []).append(cluster)
        for v in cluster:
            all_points.add("{}:{}".format(cfile, v))
    return clusters, all_points

def read_raw(filename):
    raw_messages = []
    for line in open(filename):
        line = line.strip()
        tokens = line.split()
        assert len(tokens) > 0, "Blank line in text file {}".format(filename)
        user = tokens[1]
        if tokens[0] != '===':
            user = user[1:-1]
        if len(user) == 0:
            user = tokens[2]
        raw_messages.append((user, line))

    return raw_messages

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

###    # This version will spread values out more, but also introduces a
###    # dependence between the prediction and the scale factor, which feels
###    # odd (as mentioned in the paper that proposed it).
###    normalised_VI = 1 - (I_UV / H_UV)
###    print("1 - Normalised VI:", 1 - normalised_VI)

###    # http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf
###    # Information Theoretic Measures for Clusterings Comparison: Variants,
###    # Properties, Normalization and Correction for Chance
###    # Vinh, et al. (2010), JMLR
###    #
###    #   NID = 1 - I(U, V) / max(H(U), H(V))
###    #
###    normalised_ID = 1 - (I_UV / max(H_U, H_V))
###    print("1 - Normalised ID:", 1 - normalised_ID)

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

def adjusted_mutual_information(gold, auto):
    labels_true = []
    order = []
    cur = -1
    for filename in gold:
        for gcluster in gold[filename]:
            cur += 1
            for num in gcluster:
                order.append((filename, num))
                labels_true.append(cur)
    auto_map = {}
    cur = -1
    for filename in auto:
        for acluster in auto[filename]:
            cur += 1
            for num in acluster:
                auto_map[filename, num] = cur
    labels_pred = []
    for pair in order:
        labels_pred.append(auto_map[pair])

    score = metrics.adjusted_mutual_info_score(labels_true, labels_pred, 'max')
    print("{:5.2f}   adjusted mutual information".format(100 * score))

def shen_f1(contingency, row_sums, col_sums, gold, auto):
    total = 0
    count = 0
    # For each gold cluster, calculate F-score relative to each auto cluster
    # and take the max. Then scale those scores by the size of the cluster.
    for col in col_sums:
        best = 0
        best_info = []
        col_sum = col_sums[col]
        count += col_sum
        for row in row_sums:
            n = 0
            if row in contingency and col in contingency[row]:
                n = contingency[row][col]
            if n > 0:
                row_sum = row_sums[row]
                p = n / row_sum
                r = n / col_sum
                f = 2 * p * r / (p + r)
                if f > best:
                    best = f
                    best_info = [n, row_sum, row]
        total += col_sum * best
    print("{:5.2f}   Shen F1".format(100 * total / count))

def exact_match(gold, auto, skip_single=True):
    # P/R/F over complete clusters
    total_gold = 0
    total_matched = 0
    for filename in gold:
        for cluster in gold[filename]:
            if skip_single and len(cluster) == 1:
                continue
            total_gold += 1
            matched = False
            for ocluster in auto[filename]:
                if len(ocluster.symmetric_difference(cluster)) == 0:
                    matched = True
                    break
            if matched:
                total_matched += 1
    match = []
    subsets = []
    supersets = []
    other = []
    prefix = []
    suffix = []
    gap_free = []
    match_counts = []
    subsets_counts = []
    supersets_counts = []
    other_counts = []
    prefix_counts = []
    suffix_counts = []
    gap_free_counts = []
    total_auto = 0
    for filename in auto:
        for cluster in auto[filename]:
            if skip_single and len(cluster) == 1:
                continue
            total_auto += 1
            most_overlap = 0
            fraction = 0
            count = 0
            is_subset = False
            is_superset = False
            is_prefix = False
            is_suffix = False
            is_gap_free = False
            is_match = False
            for ocluster in gold[filename]:
                if len(ocluster.symmetric_difference(cluster)) == 0:
                    is_match = True
                    break

                overlap = len(ocluster.intersection(cluster))
                if overlap > most_overlap:
                    most_overlap = overlap
                    gaps = False
                    for v in ocluster:
                        if min(cluster) <= v <= max(cluster):
                            if v not in cluster:
                                gaps = True
                    fraction = 1 - (overlap / len(ocluster.union(cluster)))
                    count = len(ocluster.union(cluster)) - overlap

                    is_subset = (overlap == len(cluster))
                    is_superset = (overlap == len(ocluster))
                    if overlap == len(cluster) and (not gaps):
                        is_gap_free = True
                        if min(ocluster) == min(cluster):
                            is_prefix = True
                        if max(ocluster) == max(cluster):
                            is_suffix = True
            if is_match:
                match.append(fraction)
                match_counts.append(count)
            elif is_superset:
                supersets.append(fraction)
                supersets_counts.append(count)
            elif is_subset:
                subsets.append(fraction)
                subsets_counts.append(count)
                if is_prefix:
                    prefix.append(fraction)
                    prefix_counts.append(count)
                elif is_suffix:
                    suffix.append(fraction)
                    suffix_counts.append(count)
                elif is_gap_free:
                    gap_free.append(fraction)
                    gap_free_counts.append(count)
            else:
                other.append(fraction)
                other_counts.append(count)
    print("Property, Proportion, Av Frac, Av Count, Max Count, Min Count")
    if len(match) > 0:
        print("Match        {:5.2f} {:5.2f} {:5.2f}".format(100 * len(match) / total_auto, 100 * sum(match) / len(match), sum(match_counts) / len(match)), max(match_counts), min(match_counts))
    if len(supersets) > 0:
        print("Super        {:5.2f} {:5.2f} {:5.2f}".format(100 * len(supersets) / total_auto, 100 * sum(supersets) / len(supersets), sum(supersets_counts) / len(supersets)), max(supersets_counts), min(supersets_counts))
    if len(subsets) > 0:
        print("Sub          {:5.2f} {:5.2f} {:5.2f}".format(100 * len(subsets) / total_auto, 100 * sum(subsets) / len(subsets), sum(subsets_counts) / len(subsets)), max(subsets_counts), min(subsets_counts))
    if len(prefix) > 0:
        print("Sub-Prefix   {:5.2f} {:5.2f} {:5.2f}".format(100 * len(prefix) / total_auto, 100 * sum(prefix) / len(prefix), sum(prefix_counts) / len(prefix)))
    if len(suffix) > 0:
        print("Sub-Suffix   {:5.2f} {:5.2f} {:5.2f}".format(100 * len(suffix) / total_auto, 100 * sum(suffix) / len(suffix), sum(suffix_counts) / len(suffix)))
    if len(gap_free) > 0:
        print("Sub-GapFree  {:5.2f} {:5.2f} {:5.2f}".format(100 * len(gap_free) / total_auto, 100 * sum(gap_free) / len(gap_free), sum(gap_free_counts) / len(gap_free)))
    if len(other) > 0:
        print("Other        {:5.2f} {:5.2f} {:5.2f}".format(100 * len(other) / total_auto, 100 * sum(other) / len(other), sum(other_counts) / len(other)))

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

def numbered_clusters(clusters):
    max_cluster = -1
    numbered = {}
    for cluster in clusters:
        max_cluster += 1
        for num in cluster:
            numbered[num] = max_cluster
    return numbered

def elsner_local_error(gold, auto, window=3):
    match = 0
    total = 0
    for filename in auto:
        anums = numbered_clusters(auto[filename])
        gnums = numbered_clusters(gold[filename])
        start = min(min(anums), min(gnums))
        end = max(max(anums), max(gnums)) + 1
        for num in range(start, end):
            if num in anums and num in gnums:
                for i in range(-window, 0):
                    pos = num + i
                    if pos in anums and pos in gnums:
                        if (gnums[pos] == gnums[num]) == (anums[pos] == anums[num]):
                            match += 1
                        total += 1
    print("{:5.2f}   Local-{}".format(100 * match / total, window))

def one_to_one(contingency, row_sums, col_sums):
    row_to_num = {}
    col_to_num = {}
    num_to_row = []
    num_to_col = []
    for row_num, row in enumerate(row_sums):
        row_to_num[row] = row_num
        num_to_row.append(row)
    for col_num, col in enumerate(col_sums):
        col_to_num[col] = col_num
        num_to_col.append(col)

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    start_nodes = []
    end_nodes = []
    capacities = []
    costs = []
    source = len(num_to_row) + len(num_to_col)
    sink = len(num_to_row) + len(num_to_col) + 1
    supplies = []
    tasks = min(len(num_to_row), len(num_to_col))
    for row, row_num in row_to_num.items():
        start_nodes.append(source)
        end_nodes.append(row_num)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    for col, col_num in col_to_num.items():
        start_nodes.append(col_num + len(num_to_row))
        end_nodes.append(sink)
        capacities.append(1)
        costs.append(0)
        supplies.append(0)
    supplies.append(tasks)
    supplies.append(-tasks)
    for row, row_num in row_to_num.items():
        for col, col_num in col_to_num.items():
            cost = 0
            if col in contingency[row]:
                cost = - contingency[row][col]
            start_nodes.append(row_num)
            end_nodes.append(col_num + len(num_to_row))
            capacities.append(1)
            costs.append(cost)

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])
  
    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow.
    min_cost_flow.Solve()

    # Score.
    total_count = sum(v for _, v in row_sums.items())
    overlap = 0
    for arc in range(min_cost_flow.NumArcs()):
        # Can ignore arcs leading out of source or into sink.
        if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:
            # Arcs in the solution have a flow value of 1. Their start and end nodes
            # give an assignment of worker to task.
            if min_cost_flow.Flow(arc) > 0:
                row_num = min_cost_flow.Tail(arc)
                col_num = min_cost_flow.Head(arc)
                col = num_to_col[col_num - len(num_to_row)]
                row = num_to_row[row_num]
                if col in contingency[row]:
                    overlap += contingency[row][col]
    print("{:5.2f}   one-to-one".format(overlap * 100 / total_count))

def correct_start(gold, auto, do_end=False):
    # P/R/F for identifying starting messages
    gold_starts = set()
    for filename in gold:
        for cluster in gold[filename]:
            start = min(cluster)
            if do_end:
                start = max(cluster)
            gold_starts.add((filename, start))

    auto_starts = set()
    for filename in auto:
        for cluster in auto[filename]:
            start = min(cluster)
            if do_end:
                start = max(cluster)
            auto_starts.add((filename, start))

    match = len(gold_starts.intersection(auto_starts))

    p = 100 * match / len(auto_starts)
    r = 100 * match / len(gold_starts)
    f = 0.0
    if match > 0:
        f = 2 * p * r / (p + r)
    prefix = "End" if do_end else "Start"
    print("{:5.2f}   {} Precision".format(p, prefix))
    print("{:5.2f}   {} Recall".format(r, prefix))
    print("{:5.2f}   {} F-score".format(f, prefix))

def start_and_end_together(gold, auto):
    # P/R/F for identifying starting messages
    total = 0
    together = 0
    for filename in gold:
        for cluster in gold[filename]:
            total += 1
            start = min(cluster)
            end = max(cluster)
            found = False
            for ocluster in auto[filename]:
                if start in ocluster and end in ocluster:
                    found = True
                    break
            if found:
                together += 1
    print("{:5.2f}   Start and end of cluster still together".format(100 * together / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate cluster / thread / conversation metrics.')
    parser.add_argument('gold', help='File containing the gold clusters, one per line. If a line contains a ":" the start is considered a filename')
    parser.add_argument('auto', help='File containing the system clusters, one per line. If a line contains a ":" the start is considered a filename')
    parser.add_argument('--metric', nargs="+", choices=['vi', 'rand', 'ami', 'shen', 'local', 'ex', 'start', '1-1', 'end', 'start-end', 'all'], default=['all'])
    args = parser.parse_args()

    gold, gpoints = read_clusters(args.gold)
    auto, apoints = read_clusters(args.auto)
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
    if 'vi' in args.metric or 'rand' in args.metric or 'shen' in args.metric or '1-1' in args.metric or 'all' in args.metric:
        contingency, row_sums, col_sums = clusters_to_contingency(gold, auto)

    if 'vi' in args.metric or 'all' in args.metric:
        variation_of_information(contingency, row_sums, col_sums)
    if 'rand' in args.metric or 'all' in args.metric:
        adjusted_rand_index(contingency, row_sums, col_sums)
    if 'ami' in args.metric or 'all' in args.metric:
        adjusted_mutual_information(gold, auto)
    if 'shen' in args.metric or 'all' in args.metric:
        shen_f1(contingency, row_sums, col_sums, gold, auto)
    if 'local' in args.metric or 'all' in args.metric:
        elsner_local_error(gold, auto)
    if 'ex' in args.metric or 'all' in args.metric:
        exact_match(gold, auto)
    if 'start' in args.metric or 'all' in args.metric:
        correct_start(gold, auto)
    if '1-1' in args.metric or 'all' in args.metric:
        one_to_one(contingency, row_sums, col_sums)
    if 'end' in args.metric or 'all' in args.metric:
        correct_start(gold, auto, True)
    if 'start-end' in args.metric or 'all' in args.metric:
        start_and_end_together(gold, auto)

