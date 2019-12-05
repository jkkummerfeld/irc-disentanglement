# Data

In this work, we annotated 77,563 messages of IRC.
Almost all are from the [Ubuntu IRC Logs](https://irclogs.ubuntu.com/) for the `#ubuntu` channel.
A small set is a re-annotation of the `#linux` channel data from [Elsner and Charniak (2008)](https://www.asc.ohio-state.edu/elsner.14/resources/chat-manual.html).

This folder contains:

File / folder      | Contents
------------------ | -----------
train              | Folder containing all training files.
dev                | Folder containing all files for development / validation.
test               | Folder containing all test files.
channel-two        | Folder containing our annotation of data from Elsner and Charniak (2008).
annotation-process | Folder containing (1) files used while developing the annotation scheme, and (2) the original annotations for dev, test, and channel-two files before adjudication.
list...txt         | Files specifying lists of files (e.g. all the training files).
glove-ubuntu.txt   | GloVe vectors, trained on all of the Ubuntu IRC logs (after tokenisation and rare word replacement with special symbols).
vocab.txt          | The vocabulary used in the GloVe vectors.
gold...txt         | Single files with all of the annotations for the train / dev / test sets.

For details about how these files were chosen for annotation and which annotator annotated each one, see [this page](./READ.history.md).

## Format

Each folder contains a set of files named as follows:

Suffix              | Contents
------------------- | -------------------------------
`.raw.txt`          | The original data from the IRC log, as downloaded.
`.ascii.txt`        | A version of the raw file that we have converted to ascii (unconvertable characters are replaced with a special word).
`.tok.txt`          | The same data agian, but with automatic tokenisation and replacement of rare words with placeholder symbols.
`.annotation.txt`   | A series of lines, each describing a link between two messages. For example: `1002 1003 -` indicates that message `1002` in the logs should be linked to message `1003`. 

Note:
- Messages are counted starting at 0 and each one is a single line in the logs.
- A message can be linked to multiple messages both before it and after it. Each link is given separately.
- A message can be linked to itself, indicating that it is the start of a new conversation.
- System messages (e.g. `=== blah has joined #ubuntu`) are counted and annotated (almost all link to themselves only).
- There are no links where both values are less than 1,000. In other words, the annotations specify what each message is a response to, starting from message 1,000.

For example, these are lines from the `2007-12-17.train-a.*` files:

```
==> data/train/2007-12-17.train-a.raw.txt <==
[03:41] <ubotu> amitprakash: To replicate your packages selection on another machine (or restore it if re-installing), you can type « dpkg --get-selections > ~/my-packages », move the file "my-packages" to the other machine, and there type « sudo dpkg --set-selections < my-packages && sudo apt-get dselect-upgrade » - See also !automate
[03:41] <hwilde> speeddemon8803, ever run ifconfig and lo is missing?
[03:42] <scguy318> darklordveynom: since runlevels are supplanted by Upstart

==> data/train/2007-12-17.train-a.ascii.txt <==
[03:41] <ubotu> amitprakash: To replicate your packages selection on another machine (or restore it if re-installing), you can type <unconvertable> dpkg --get-selections > ~/my-packages <unconvertable> , move the file "my-packages" to the other machine, and there type <unconvertable> sudo dpkg --set-selections < my-packages && sudo apt-get dselect-upgrade <unconvertable> - See also !automate
[03:41] <hwilde> speeddemon8803, ever run ifconfig and lo is missing?
[03:42] <scguy318> darklordveynom: since runlevels are supplanted by Upstart

==> data/train/2007-12-17.train-a.tok.txt <==
<s> <user> : to replicate your packages selection on another machine ( or restore it if re-installing ), you can type <unconvertable > dpkg --get-selections > DIR/~/ DIR/my-packages/ <unconvertable > , move the file " my-packages " to the other machine , and there type <unconvertable > sudo dpkg --set-selections < my-packages && sudo apt-get dselect-upgrade <unconvertable > - see also ! automate </s>
<s> <user> , ever run ifconfig and lo is missing ? </s>
<s> <user> : since runlevels are <unka> by upstart </s>

==> data/train/2007-12-17.train-a.annotation.txt <==
993 1000 -
1000 1001 -
1002 1002 -
```

Tokenisation was performed using the script in the tools directory.

[Go back](./../) to the main webpage.
