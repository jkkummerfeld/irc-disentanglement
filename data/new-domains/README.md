This folder contains files from the paper [Chat Disentanglement: Data for New Domains and Methods for More Accurate Annotation](), by Sai R. Gouravajhala, Andrew M. Vernier, Yiming Shi, Zihan Li, Mark Ackerman, Jonathan K. Kummerfeld.

The 24 files provide annotations for 2,400 lines of annotated IRC data, and 12,000 for context:

- 4 channels
- 3 samples from each channel
- 200 lines of annotated data, with 1,000 preceding lines as context.

The format is the same as for the earlier data. However, these IRC channels have slightly different logs:

Channel | Example
------- | -------
Mediawiki | `mediawiki 2013-01-26 [15:04:04] <wikibugs>	 03(mod) Wrong escaping in check warning - 10https://bugzilla.wikimedia.org/44381  +comment (10niklas.laxstrom)`
Rust | `rust 2018-05-29 [21:20:37] <talchas> but I don't know that I'd bother`
Stripe | `stripe 2019-09-04 [22:44:46] <w1zeman1p> If the customer was created < 1.month.ago, then add a coupon when you create the subscription`
Ubuntu Meeting | `ubuntu-meeting 2010-11-08 [13:21] <rodrigo_> the new gnome-control-center panel?`

If you use this data in your work, please cite it as:

```
@InProceedings{alta23disentangle,
  author    = {Sai R. Gouravajhala and Andrew M. Vernier and Yiming Shi and Zihan Li and Mark Ackerman and Jonathan K. Kummerfeld},
  title     = {Chat Disentanglement: Data for New Domains and Methods for More Accurate Annotation},
  booktitle = {Proceedings of the The 21st Annual Workshop of the Australasian Language Technology Association},
  location  = {Melbourne, Australia},
  month     = {November},
  year      = {2023},
  doi       = {},
  pages     = {},
  url       = {},
  arxiv     = {},
  data      = {https://www.jkk.name/irc-disentanglement},
}
```

[go back](./../../) to the main webpage.
