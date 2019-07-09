This is an old version of the code that is not as effective.
It is included for completeness.

## Compile

1. Install DyNet for C++ as described [here](https://dynet.readthedocs.io/en/latest/install.html).
2. Run `make` in this directory.

## Train and run

This command will run the system assuming you have compiled it and are now in the root directory of this repository:

```
./src/old-cpp-version/predictor
  --dynet-mem 3048
  --dynet-autobatch 1
  --dynet-weight-decay 0.00000001
  --data-train data/list.ubuntu.train.txt
  --data-dev data/list.ubuntu.dev.txt
  --data-eval data/list.ubuntu.test.txt
  --log-freq 5000
  --dev-freq 1
  --trainer sgd
  --dim-input 100
  --model ff
  --prefix example-run-cpp
  --instance-type selection
  --learning-rate 0.1
  --selection-proportion 0.1
  --input-hand-crafted
  --nonlinearity tanh
  --input-text
  --dim-ff-hidden 64
  --layers-ff-pair 2
  --max-link-length 100
  --max-iterations 200
  --word-vector-init ubuntu-all.tok-user.unked-100.w2v.txt
  --no-improvement-cutoff 30
  --clipping-threshold 1.0
```

