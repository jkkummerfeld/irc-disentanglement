## Evaluation

We provide three scripts for evaluation.

### Graph Evaluation

This calculates precision, recall, and F-score over edges.

```
python3 tools/evaluation/graph-eval.py --gold gold-files --auto system-files
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

### Conversation Evaluation

This calculates a range of cluster metrics.
It requires the [Google OR Tools](https://developers.google.com/optimization/install/python/).

```
python3 tools/evaluation/conversation-eval.py gold-file system-file
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

### DSTC 8 Evaluation

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

The expected format is the same as for `graph-eval.py`

### Annotator Agreement

This is the code we used to calculate agreement between annotators.
It expects to receive a list of filenames as arguments, where files are named:

```
DATA.annotation.ANNOTATOR
```

Results are printed for each pair of annotators.

### Significance Testing

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

### Download IRC data

`./download-raw-data.sh` will download all available IRC data from the Ubuntu channel.
Note, this includes some days in which there is no data or the files have other issues.

### Convert to ASCII

We converted downloaded files to ascii in two steps:

1. Run `make-txt.py` to convert as much as possible (this will handle most cases of things like accents).
2. Use `tr -cd '[:print:]\\n\\t' < filename > filename.ascii` to delete remaining non-ascii characters.

### Tokenise

We have two tokenisation scripts (preliminary experiments with off-the-shelf tokenisers gave poor results).
The easiest one to run is the one we've prepared for DSTC 8:

```
./dstc8-tokenise.py --vocab vocab.txt --output-suffix .tok ASCII-FILENAME
```

The other (`tokenise.py`) has additional options that are generaly not needed.

Run to convert text to tokens with unk symbols.

[Go back](./../) to the main webpage.
