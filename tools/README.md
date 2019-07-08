evaluation
format_conversion
preprocessing

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

