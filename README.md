# irc-disentanglement
This repository contains data and code for disentangling conversations on IRC, as described in:

  - [A Large-Scale Corpus for Conversation Disentanglement](https://aclweb.org/anthology/papers/P/P19/P19-1374/),
  Jonathan K. Kummerfeld, Sai R. Gouravajhala, Joseph Peper, Vignesh Athreya, Chulaka Gunasekara, Jatin Ganhotra, Siva Sankalp Patel, Lazaros Polymenakos, and Walter S. Lasecki,
  ACL 2019

Conversation disentanglement is the task of identifying separate conversations in a single stream of messages.
For example, the image below shows two entangled conversations and an annotated graph structure (indicated by lines and colours).
The example includes a message that receives multiple responses, when multiple people independently help BurgerMann, and the inverse, when the last message responds to multiple messages.
We also see two of the users, delire and Seveas, simultaneously participating in two conversations.

<img src="https://raw.githubusercontent.com/jkkummerfeld/irc-disentanglement/master/example-conversation.png" width="500" alt="Image of an IRC message log with conversations marked">

This work:

1. Introduces a new dataset, with disentanglement for 77,563 messages of IRC.
2. Introduces a new model, which achieves significantly higher results than prior work.
3. Re-analyses prior work, identifying issues with data and assumptions in models.

To get our code and data, download this repository in one of these ways:

- [Download .zip](https://github.com/jkkummerfeld/irc-disentanglement/zipball/master)
- [Download .tar.gz](https://github.com/jkkummerfeld/irc-disentanglement/tarball/master)
- `git clone https://github.com/jkkummerfeld/irc-disentanglement.git`

The data is also available here:

- [huggingface datasets](https://huggingface.co/datasets/irc_disentangle)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/irc_disentanglement)

This repository contains:

- The [annotated data](./data) for both Ubuntu and Channel Two.
- The code for our [model](./src/).
- The code for [tools](./tools/) that do evaluation, preprocessing and data format conversion.
- A collection of 496,469 automatically disentangled conversations from 2004 to 2019 in a [bzip2 file](./acl19-irc-disentanglement_auto-data-full.txt.bz2).

If you use the data or code in your work, please cite our work as:

```
@InProceedings{acl19disentangle,
  author    = {Jonathan K. Kummerfeld and Sai R. Gouravajhala and Joseph Peper and Vignesh Athreya and Chulaka Gunasekara and Jatin Ganhotra and Siva Sankalp Patel and Lazaros Polymenakos and Walter S. Lasecki},
  title     = {A Large-Scale Corpus for Conversation Disentanglement},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  location  = {Florence, Italy},
  month     = {July},
  year      = {2019},
  doi       = {10.18653/v1/P19-1374},
  pages     = {3846--3856},
  url       = {https://aclweb.org/anthology/papers/P/P19/P19-1374/},
  arxiv     = {https://arxiv.org/abs/1810.11118},
  software  = {https://jkk.name/irc-disentanglement},
  data      = {https://jkk.name/irc-disentanglement},
}
```

# Running and Reproducing Results

See [the src folder README](./src/) for detailed instructions on running the system.
Additional evaluation script information can be found in the [tools README](./tools/).

# Updates

1. The description of the voting ensemble in the paper has a mistake. When not all models agree, the most agreed upon link is chosen (ties are broken by choosing the shorter link).

# Questions

If you have a question please either:

- Open an issue on [github](https://github.com/jkkummerfeld/irc-disentanglement/issues).
- Mail me at [jkummerf@umich.edu](mailto:jkummerf@umich.edu).

# Contributions

If you find a bug in the data or code, please submit an issue, or even better, a pull request with a fix.
I will be merging fixes into a development branch and only infrequently merging all of those changes into the master branch (at which point this page will be adjusted to note that it is a new release).
This approach is intended to balance the need for clear comparisons between systems, while also improving the data.

# Acknowledgments

This material is based in part upon work supported by IBM under contract 4915012629.
Any opinions, findings, conclusions or recommendations expressed are those of the authors and do not necessarily reflect the views of IBM.
