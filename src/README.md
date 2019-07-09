# System

This folder contains code for reproducing our disentanglement experiments.

## Requirements

The only dependency is the [DyNet library](dynet.readthedocs.io), which can usually be installed with:

```
pip3 install dynet
```

## Running

To see all options, run:

```
python3 disentangle.py --help
```

### Train

To train, provide the `--train` argument followed by a series of filenames.

The example command below will train a model with the same parameters as used in the ACL paper.
The model is a feedforward neural network with 2 layers, 512 dimensional hidden vectors, and softsign non-linearities.

```
python3 disentangle.py example-train \
  --train ../data/train/*annotation.txt \
  --dev ../data/dev/*annotation.txt \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  --epochs 20 \
  --dynet-autobatch \
  --drop 0 \
  --learning-rate 0.018804 \
  --learning-decay-rate 0.103 \
  --seed 10 \
  --clip 3.740 \
  --weight-decay 1e-07 \
  --opt sgd \
  > example-train.out 2>example-train.err
```

### Infer

This command will run the model trained above on the development set:

```
python3 disentangle.py sample-run \
  --model sample-train.dy.model \
  --test ../data/dev/*annotation* \
  --test-start 1000 \
  --test-end 2000 \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  > sample-run.out 2>sample-run.err
```

Note - the arguments defining the network (hiiden, layers, nonlin), must match those given in training.

## Ensemble

For the best results, we used a simple ensemble of multiple models.
We trained 10 models as described above, but with different random seeds.
We combined their output using the `majority_vote.py` script in this directory.

The same script is used for all three ensemble methods, with slightly different input and arguments:

Union
```
ls output*graphs | ./majority_vote.py 1 > output.combined.union
```

Vote
```
ls output*graphs | ./majority_vote.py 10 > output.combined.vote
```

Intersect
```
ls output*clusters | ./majority_vote.py 10 > output.combined.intersect
```

All of these assume the output files have been converted into our graph format, like this:
```
for name in output*out ; do ../tools/format-conversion/output-from-cpp-to-graph.py < $name > $name.graphs ; done
```

The intersect method also assumes they have been made into clusters, like this:
```
for name in output*out ; do ../tools/format-conversion/graph-to-cluster.py < $name.graphs > $name.clusters ; done
```

## C++ Model

As well as the main Python code, we also wrote a model in C++ that was used for DSTC 7 and the results in the 2018 arXiv version of the paper (the Python version was used for DSTC 8 and the 2019 ACL paper).
The python model has additional input features and a different text representation method.
The C++ model has support for a range of additional variations in both inference and modeling, which did not appear to improve performance.
For details on how to build and run the C++ code, see [this page](./old-cpp-version/).

[Go back](./../) to the root of the repository.
