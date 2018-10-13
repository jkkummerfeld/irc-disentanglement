Dataset and model for disentangling chat on IRC

# Data

Total lines labeled:
- 18,924 from 48 hr x 1   (train 1)
- 1,000 from 100 x 10 x 3 (test 1)
- 2,500 from 250 x 10 x 2 (dev 2)
- 5,000 from 500 x 10 x 3 (test 2)
- 2,600 from 2600 x 1 x 2 (elsner 2)
- ?     from 500 x ? x 1  (train 2)
30,024 + ? labeled
47,124 + ? labels

Training chosen by randomly picking start messages and keeping 1500 after that.
Removed cases with '=== ...' messages dominating or in a huge clump.
Provided 1,000 messages of context.

# Code

## Build assumptions:

- Installed DyNet, Eigen
- Defined environment variables for DYNET and EIGEN, (MKL threads if in use), and the C++ compiler, for example:

In `~/.bash_profile`

```
export DYNET=/path/to/dynet
export EIGEN=/path/to/eigen
export MKL_NUM_THREADS=2
export CXX=g++
```

Where the paths are to the root directory of the source, and I assume their make processes were run with their default setup.

## Running code

Example run:

```
./bin/predictor --dynet-mem 3048 --dynet-autobatch 1 --data-train data/list.quarter.train.splitNA.txt --data-dev data/list.quarter.dev.splitNA.txt --data-eval data/list.quarter.dev.splitNA.txt --log-freq 4000 --dev-freq 1 --trainer sgd --model ff --prefix scratch/feb17.ff.quarter.selection.selu.0.0.0.0 --nonlinearity selu --dropout-input 0.0 --dropout-ff 0.0 --instance-type selection --input-hand-crafted --learning-rate 0.0015 --loss-type kHinge --context-size 1
```

## Data preprocessing

The following preprocessing steps were applied to the training and test data:

```bash
# Convert to ascii
tools/preprocessing/make_txt.py ${filename}

# Tokenise and replace unknown tokens
tools/preprocessing/tokenise.py --replace_usernames --cut_timestamp --cut_username --add_line_boundaries --use_vocab ../train/vocab.100 <filename>
```

To get the `vocab.100` file, and to process the complete logs, we did the following:

```bash
# Download data
tools/preprocessing/get_raw_data/download.sh

# Merged and added date stamps
grep '.' 20*/*/* > ubuntu-full.raw.txt

# Convert to ascii
tools/preprocessing/make_txt.py ubuntu-full.raw.txt > ubuntu-full.ascii.txt

# Tokenise
tools/preprocessing/tokenise.py --replace_usernames --cut_timestamp --cut_username --add_line_boundaries ubuntu-full.ascii.txt > ubuntu-full.ascii.tok

# Get frequent tokens
tools/preprocessing/count_unique.py < ubuntu-full.ascii.tok > ubuntu-full.ascii.tok.counts
awk '$1 > 99' ubuntu-full.ascii.tok.counts > ubuntu-full.ascii.tok.counts-100

# Tokenise again, now replacing rare words with unknown tokens
tools/preprocessing/tokenise.py --replace_usernames --cut_timestamp --cut_username --add_line_boundaries --use_vocab ubuntu-full.ascii.tok.counts-100 ubuntu-full.ascii.txt > ubuntu-full.ascii.tok.unk-100
```

