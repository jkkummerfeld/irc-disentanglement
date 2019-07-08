# Disentanglement Baseline and Evaluation

This folder contains the code needed to run and evaluate the baseline disentanglement model. The files are:

- `README.md`, this file
- `task-4-evaluation.py`, the evaluation program
- `disentangle.py`, the baseline
- `reserved_words.py`, a set of words used by the baseline (kept in a separate file for convenience)
- `disentangle.model`, a trained model for the baseline
- `glove-ubuntu.txt`, a set of word vectors trained on all of the Ubuntu data (after applying our tokenisation method)
- `tokenise.py`, a tool to tokenise the text ~as we have in the provided files

## Baseline

The baseline is a feedforward neural network with 2 layers, 512 dimensional hidden vectors, and softsign non-linearities. It uses the DyNet library:

dynet.readthedocs.io

This can usually be installed with:

`pip3 install dynet`

This will train a model with the same parameters as we have used here:

```
python3 disentangle.py example-train --train ../task-4/train/*annotation.txt --dev ../task-4/dev/*annotation.txt --hidden 512 --drop 0 --layers 2 --nonlin softsign --word-vectors glove-ubuntu.txt --epochs 20 --dynet-autobatch --learning-rate 0.018804 --learning-decay-rate 0.103 --seed 10 --clip 3.740 --weight-decay 1e-07 --momentum 0.96 --opt sgd > example-train.out 2>example-train.err
```

This will run the trained model on the development set:

```
python3 disentangle.py example-run --model disentangle.model --test ../task-4/dev/*annotation* --hidden 512 --drop 0 --layers 2 --nonlin softsign --test-start 1000 --test-end 2000 --word-vectors glove-ubuntu.txt > example-run.out 2>example-run.err
```

For a full list of arguments run:

```
python3 disentangle.py --help
```

## Evaluation

To evaluate the output of a system use:

```
python3 task-4-evaluation.py --gold ../task-4/dev/*anno* --auto example-run.out
```

For the provided model, that should give:

```
92.15   1 - Scaled VI
74.19   Adjusted rand index
40.53   Matched clusters precision
41.26   Matched clusters recall
40.89   Matched clusters f-score
```

The format for output files is:

```
anything.../filename:line_number line_number -
```

For example:

```
../task-4/dev/2004-11-15.annotation.txt:1000 1000 -
../task-4/dev/2004-11-15.annotation.txt:1001 1001 -
../task-4/dev/2004-11-15.annotation.txt:1002 1002 -
../task-4/dev/2004-11-15.annotation.txt:1003 1002 -
../task-4/dev/2004-11-15.annotation.txt:1004 1003 -
../task-4/dev/2004-11-15.annotation.txt:1005 1003 -
```

## Tokenise

Run to convert text to tokens with unk symbols. NOTE: we have done this for you on all the provided files in task 4. This is mainly here in case you wish to use the same process on task 1 or task 2.

```
./tokenise.py --vocab vocab.txt --output-suffix .tok ASCII_FILENAME
```

