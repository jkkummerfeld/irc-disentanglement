#!/bin/bash

DIS_HOME="Put the base of the repository here, e.g. /home/jkkummerfeld/disentangle/"
tmpfile=todo.$RANDOM
cat - > ${tmpfile}.ascii.txt

# Assuming you use anaconda for package management and have set up an environment called 'dynet'
source "Location of anaconda"/etc/profile.d/conda.sh
conda activate dynet

python3 ${DIS_HOME}/tools/preprocessing/dstc8-tokenise.py --vocab ${DIS_HOME}/data/vocab.txt --output-suffix .tok ${tmpfile}.ascii.txt
mv ${tmpfile}.ascii.txt.tok ${tmpfile}.tok.txt

python3 ${DIS_HOME}/src/disentangle.py \
  ${tmpfile} \
  --model ${DIS_HOME}/disentanglement-model.dy.model \
  --test ${tmpfile}.ascii.txt \
  --test-start 0 \
  --test-end 10000000 \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ${DIS_HOME}/data/glove-ubuntu.txt \
  > ${tmpfile}.out 2>${tmpfile}.err

cat ${tmpfile}.out | grep -v '^[#]' | sed 's/.*[:]//'
