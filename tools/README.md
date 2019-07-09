
## Evaluation

We provide three scripts for evaluation.

### graph-eval.py

This calculates precision, recall, and F-score over edges.

### thread-eval.py

This calculates a range of graph metrics.
It requires the [Google OR Tools](https://developers.google.com/optimization/install/python/).

```
python3 tools/evaluation/thread-eval.py gold-file system-file
```

The expected format for output files has each cluster on one line:

```
anything.../filename:line-number line-number line-number line-number ...
```

For example:

```
blah-blah/2004-11-15.annotation.txt:1000
blah-blah/2004-11-15.annotation.txt:1001
yaddah_yaddah/2004-11-15.annotation.txt:1002 1003 1004 1005
```

It also has the option to specify the set of metrics to use.

### dstc8-evaluation.py

To evaluate the output of a system use:
This is the code being used in DSTC 8.
It is run with:

```
python3 tools/evaluation/dstc8-evaluation.py --gold data/dev/*anno* --auto example-run.out
```

For a model trained as described in the source code README, that should give:

```
92.15   1 - Scaled VI
74.19   Adjusted rand index
40.53   Matched clusters precision
41.26   Matched clusters recall
40.89   Matched clusters f-score
```

The expected format for output files is:

```
anything.../filename:line-number line-number -
```

For example:

```
blah-blah/2004-11-15.annotation.txt:1000 1000 -
blah-blah/2004-11-15.annotation.txt:1001 1001 -
blah-blah/2004-11-15.annotation.txt:1002 1002 -
blah-blah/2004-11-15.annotation.txt:1003 1002 -
yaddah_yaddah/2004-11-15.annotation.txt:1004 1003 -
yaddah_yaddah/2004-11-15.annotation.txt:1005 1003 -
```

### agreement.py

This is the code we used to calculate agreement between annotators.
It expects to receive a list of filenames as arguments, where files are named:

```
DATA.annotation.ANNOTATOR
```

Results are printed for each pair of annotators.

### significance.py

This uses a permutation test, comparing two sets of annotations.
It is run with:

```
./tools/evaluation/significance.py \
  gold-file
  --auto0 system0-output-files
  --auto system1-output-files
```

## Format Conversion

These are a set of tools for converting annotations:

Tool                             | Purpose
-------------------------------- | ---------------
output-from-py-to-graph.py       | Convert the output of the model into our graph format.
graph-to-cluster.py              | Convert a set of links into a set of conversations with Union-Find.
cluster-to-messages.py           | Print out the actual messages that a set of conversations correspond to.
graph-to-messages.py             | Do both graph-to-cluster and cluster-to-message at once.
output-from-cpp-to-graph.py      | Convert the output of the C++ model into our graph format.

## Preprocessing

Tokenise

Run to convert text to tokens with unk symbols. NOTE: we have done this for you on all the provided files in task 4. This is mainly here in case you wish to use the same process on task 1 or task 2.

```
./tokenise.py --vocab vocab.txt --output-suffix .tok ASCII-FILENAME
```

