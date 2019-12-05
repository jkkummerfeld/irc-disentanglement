# Annotation Information

The data was selected and annotated in various ways.
The [supplementary material to the paper](https://github.com/jkkummerfeld/irc-disentanglement/blob/master/supp-acl19irc.pdf) discusses these, and below we provide additional details.

## Training Set

Each file was annotated by one person. All annotators had gone through the training process.

### Part A

Selected by:
1. Calculate stats for every hour
2. Determine cutoffs for 0-25%, 25-50%, 50-75%, and 75-100% on each axis (users, messages, directed)
3. For each case, divide the data into four, and select four hours from each section

Note, the identification of directed messages used an earlier version of our code that was less accurate.
However, the difference is probably small (1-5%).

Cutoffs were:
- Users: 23, 38, 56
- Messages: 145, 264, 405
- Directed: 36.0, 43.8, 50.6

Stats used:
- Users 0 2013-10-11.txt:07am u:19 m:76 %:39
- Users 0 2012-11-24.txt:20am u:21 m:211 %:45
- Users 0 2013-07-10.txt:23am u:12 m:131 %:46
- Users 0 2013-09-12.txt:11am u:19 m:253 %:43
- Users 25 2013-12-02.txt:21am u:32 m:170 %:35
- Users 25 2011-08-17.txt:08am u:24 m:132 %:34
- Users 25 2012-12-15.txt:02am u:28 m:165 %:42
- Users 25 2005-05-19.txt:11am u:23 m:145 %:20
- Users 50 2009-03-25.txt:11am u:40 m:344 %:40
- Users 50 2010-02-13.txt:01am u:43 m:311 %:35
- Users 50 2009-01-05.txt:11am u:48 m:308 %:44
- Users 50 2008-04-20.txt:09am u:48 m:373 %:50
- Users 75 2006-06-05.txt:12pm u:77 m:600 %:44
- Users 75 2011-04-28.txt:09am u:69 m:405 %:46
- Users 75 2006-12-10.txt:09pm u:71 m:492 %:49
- Users 75 2006-06-01.txt:12pm u:129 m:952 %:45
- Messages 0 2013-08-30.txt:11am u:22 m:126 %:60
- Messages 0 2013-01-11.txt:13am u:24 m:68 %:35
- Messages 0 2012-06-02.txt:18am u:35 m:134 %:37
- Messages 0 2013-10-04.txt:12am u:16 m:58 %:18
- Messages 25 2013-05-19.txt:22am u:34 m:175 %:53
- Messages 25 2005-02-08.txt:02pm u:24 m:174 %:38
- Messages 25 2012-05-04.txt:04am u:22 m:205 %:60
- Messages 25 2006-02-24.txt:10am u:38 m:253 %:50
- Messages 50 2010-08-05.txt:11am u:49 m:271 %:42
- Messages 50 2012-11-30.txt:14am u:35 m:279 %:55
- Messages 50 2013-02-24.txt:02am u:31 m:275 %:54
- Messages 50 2013-01-30.txt:01am u:24 m:280 %:36
- Messages 75 2007-09-07.txt:07pm u:85 m:534 %:49
- Messages 75 2008-06-03.txt:11am u:44 m:429 %:40
- Messages 75 2007-06-04.txt:10pm u:82 m:650 %:45
- Messages 75 2007-01-19.txt:07am u:50 m:629 %:56
- Directed 0 2013-07-19.txt:03am u:16 m:190 %:27
- Directed 0 2007-07-03.txt:06pm u:82 m:573 %:35
- Directed 0 2013-05-05.txt:18am u:35 m:157 %:31
- Directed 0 2013-10-28.txt:03am u:25 m:294 %:27
- Directed 25 2007-12-17.txt:04am u:75 m:641 %:41
- Directed 25 2005-07-25.txt:11am u:35 m:210 %:39
- Directed 25 2006-08-23.txt:03am u:69 m:542 %:43
- Directed 25 2012-05-20.txt:16am u:54 m:309 %:43
- Directed 50 2007-06-17.txt:06pm u:85 m:564 %:45
- Directed 50 2007-08-19.txt:01am u:73 m:517 %:43
- Directed 50 2012-02-03.txt:20am u:42 m:230 %:50
- Directed 50 2008-04-30.txt:08am u:65 m:696 %:46
- Directed 75 2008-04-27.txt:06am u:93 m:969 %:55
- Directed 75 2005-06-20.txt:03pm u:32 m:359 %:52
- Directed 75 2010-08-29.txt:20am u:58 m:423 %:59
- Directed 75 2013-05-07.txt:02am u:17 m:144 %:63

### Part B

Chosen by:
1. Filter out all hours that are in the max or min 5% for users, messages, or addressing
2. Randomly select 10 hours
3. For busy hours, keep the first 100 messgaes, for quiet hours, add messages from the following hour to get up to 100

- All 0 2015-11-26.txt:10am u:16 m:54 %:37
- All 0 2015-06-12.txt:20am u:19 m:83 %:34
- All 0 2015-09-25.txt:19am u:25 m:108 %:46
- All 0 2015-04-19.txt:18am u:17 m:182 %:41
- All 0 2015-10-19.txt:16am u:23 m:156 %:53
- All 0 2015-01-20.txt:04am u:13 m:157 %:54
- All 0 2015-08-10.txt:08am u:15 m:118 %:27
- All 0 2015-12-28.txt:16am u:13 m:71 %:33
- All 0 2015-02-04.txt:15am u:16 m:83 %:36
- All 0 2015-10-14.txt:21am u:15 m:155 %:25

### Part C

Selected by choosing a random point in the logs and keeping 1,500 messages after that point (1,000 as context, 500 to annotate).

```
2004-12-25 2005-02-06 2005-02-27 2005-05-14 2005-06-06 2005-06-12 2005-06-16
2005-07-29 2005-09-26 2005-10-07 2005-10-12 2005-12-03 2005-12-04 2005-12-16
2005-12-23 2006-01-02 2006-01-12 2006-02-20 2006-02-28 2006-03-05 2006-05-02
2006-05-15 2006-05-27 2006-05-29 2006-06-08 2006-06-21 2006-06-28 2006-07-01
2006-08-06 2006-08-11 2006-08-13 2006-08-15 2006-09-13 2006-09-24 2006-11-01
2006-12-06 2006-12-20 2007-01-12 2007-01-21 2007-01-29 2007-02-06 2007-02-07
2007-02-15 2007-06-01 2007-08-22 2007-08-24 2007-10-24 2008-01-02 2008-01-03
2008-02-07 2008-02-14 2008-03-01 2008-05-24 2008-07-03 2008-10-02 2009-05-04
2009-05-08 2009-07-02 2009-11-13 2009-12-05 2010-01-04 2010-03-08 2010-03-20
2010-04-12 2010-05-30 2010-06-21 2010-08-15 2010-10-17 2010-10-27 2011-02-13
2011-02-23 2011-03-18 2011-04-14 2011-04-17 2011-08-22 2011-11-24 2011-12-07
2012-03-24 2012-06-20 2013-05-28 2013-08-29 2013-09-16 2014-01-08 2014-08-14
2014-09-29 2014-12-21 2014-12-27 2015-05-08 2017-02-06 2017-03-02 2017-03-23
2017-05-09 2017-07-15 2017-09-02 2018-02-27
```

## Development Set

Selected by choosing a random point in the logs and keeping 1,250 messages after that point (1,000 as context, 250 to annotate).

```
2004-11-15_03 2005-06-27_12 2005-08-08_01 2008-12-11_11 2009-02-23_10
2009-03-03_10 2009-10-01_17 2011-05-29_19 2011-11-13_02 2016-12-19_20
```

## Test Set

Selected by choosing a random point in the logs and keeping 1,500 messages after that point (1,000 as context, 500 to annotate).

```
2005-07-06_14 2007-01-11_12 2007-12-01_03 2008-07-14_18 2010-08-17_18
2013-09-01_02 2014-06-18_13 2015-03-18_05 2016-02-22_17 2016-06-08_07
```

## Pilot Data

Used in the process of developing the annotation scheme, **NOT intended for use in developing or evaluating models**.
If you use this data for either training or tuning your model your results with NOT be comparable with those in the paper.
This is included mainly for completeness.

Overall 1,250 lines (counting is a little subtle as it includes lines that didn't get a label)

Phase 1:
- Development round 1
  - 2016-11-01_00.annotation.jonathan.txt
  - 2016-11-02_00.annotation.jonathan.txt

- Development round 2
  - 2005-04-05_10.annotation.jonathan.txt
  - 2005-04-05_10.annotation.joseph.txt
  - 2006-03-15_03.annotation.jonathan.txt
  - 2006-03-15_03.annotation.joseph.txt
  - 2006-11-30_08.annotation.jonathan.txt
  - 2006-11-30_08.annotation.joseph.txt

- Development round 3
  - 2016-06-06_02.annotation.jonathan.txt
  - 2016-06-06_02.annotation.joseph.txt
  - 2016-06-06_18.annotation.jonathan.txt
  - 2016-06-06_18.annotation.joseph.txt

- Training additional annotator
  - 2005-04-05_10.annotation.vignesh.txt
  - 2006-03-15_03.annotation.vignesh.txt
  - 2006-03-15_03.annotation.mturk.txt - not used
  - 2006-03-15_03.annotation.rui.txt - not used

- Agreement check (end):
  - 2015-01-20_04.annotation.jonathan.txt
  - 2015-01-20_04.annotation.joseph.txt
  - 2015-01-20_04.annotation.vignesh.txt
  - 2015-02-04_15.annotation.jonathan.txt
  - 2015-02-04_15.annotation.joseph.txt
  - 2015-02-04_15.annotation.vignesh.txt

Phase 2:
- Training new annotators:
  - 2015-01-20_04.annotation.hussam.txt
  - 2015-01-20_04.annotation.jared.txt
  - 2015-02-04_15.annotation.hussam.txt
  - 2015-02-04_15.annotation.jared.txt

- Agreement check (half way through):
  - 2006-03-15_03.annotation.hussam.txt
  - 2006-03-15_03.annotation.jared.txt


- Annotator records:
    - train/2004-12-25.train-c.annotation.txt    jared
    - train/2005-02-06.train-c.annotation.txt    hussam
    - train/2005-02-08.train-a.annotation.txt    vignesh
    - train/2005-02-27.train-c.annotation.txt    jared
    - train/2005-05-14.train-c.annotation.txt    hussam
    - train/2005-05-19.train-a.annotation.txt    vignesh
    - train/2005-06-06.train-c.annotation.txt    hussam
    - train/2005-06-12.train-c.annotation.txt    hussam
    - train/2005-06-16.train-c.annotation.txt    hussam
    - train/2005-06-20.train-a.annotation.txt    jonathan
    - train/2005-07-25.train-a.annotation.txt    vignesh
    - train/2005-07-29.train-c.annotation.txt    jared
    - train/2005-09-26.train-c.annotation.txt    jared
    - train/2005-10-07.train-c.annotation.txt    hussam
    - train/2005-10-12.train-c.annotation.txt    hussam
    - train/2005-12-03.train-c.annotation.txt    hussam
    - train/2005-12-04.train-c.annotation.txt    jared
    - train/2005-12-16.train-c.annotation.txt    hussam
    - train/2005-12-23.train-c.annotation.txt    jared
    - train/2006-01-02.train-c.annotation.txt    jared
    - train/2006-01-12.train-c.annotation.txt    hussam
    - train/2006-02-20.train-c.annotation.txt    hussam
    - train/2006-02-24.train-a.annotation.txt    vignesh
    - train/2006-02-28.train-c.annotation.txt    jared
    - train/2006-03-05.train-c.annotation.txt    hussam
    - train/2006-05-02.train-c.annotation.txt    jared
    - train/2006-05-15.train-c.annotation.txt    jared
    - train/2006-05-27.train-c.annotation.txt    jared
    - train/2006-05-29.train-c.annotation.txt    jared
    - train/2006-06-01.train-a.annotation.txt    joseph
    - train/2006-06-05.train-a.annotation.txt    joseph
    - train/2006-06-08.train-c.annotation.txt    hussam
    - train/2006-06-21.train-c.annotation.txt    jared
    - train/2006-06-28.train-c.annotation.txt    hussam
    - train/2006-07-01.train-c.annotation.txt    jared
    - train/2006-08-06.train-c.annotation.txt    jared
    - train/2006-08-11.train-c.annotation.txt    jared
    - train/2006-08-13.train-c.annotation.txt    hussam
    - train/2006-08-15.train-c.annotation.txt    jared
    - train/2006-08-23.train-a.annotation.txt    jonathan
    - train/2006-09-13.train-c.annotation.txt    jared
    - train/2006-09-24.train-c.annotation.txt    hussam
    - train/2006-11-01.train-c.annotation.txt    jared
    - train/2006-12-06.train-c.annotation.txt    jared
    - train/2006-12-10.train-a.annotation.txt    joseph
    - train/2006-12-20.train-c.annotation.txt    jared
    - train/2007-01-12.train-c.annotation.txt    jared
    - train/2007-01-19.train-a.annotation.txt    jonathan
    - train/2007-01-21.train-c.annotation.txt    jared
    - train/2007-01-29.train-c.annotation.txt    jared
    - train/2007-02-06.train-c.annotation.txt    jared
    - train/2007-02-07.train-c.annotation.txt    hussam
    - train/2007-02-15.train-c.annotation.txt    hussam
    - train/2007-06-01.train-c.annotation.txt    hussam
    - train/2007-06-04.train-a.annotation.txt    vignesh
    - train/2007-06-17.train-a.annotation.txt    joseph
    - train/2007-07-03.train-a.annotation.txt    joseph
    - train/2007-08-19.train-a.annotation.txt    vignesh
    - train/2007-08-22.train-c.annotation.txt    hussam
    - train/2007-08-24.train-c.annotation.txt    hussam
    - train/2007-09-07.train-a.annotation.txt    jonathan
    - train/2007-10-24.train-c.annotation.txt    hussam
    - train/2007-12-17.train-a.annotation.txt    joseph
    - train/2008-01-02.train-c.annotation.txt    jared
    - train/2008-01-03.train-c.annotation.txt    jared
    - train/2008-02-07.train-c.annotation.txt    jared
    - train/2008-02-14.train-c.annotation.txt    jared
    - train/2008-03-01.train-c.annotation.txt    hussam
    - train/2008-04-20.train-a.annotation.txt    jonathan
    - train/2008-04-27.train-a.annotation.txt    vignesh
    - train/2008-04-30.train-a.annotation.txt    jonathan
    - train/2008-05-24.train-c.annotation.txt    jared
    - train/2008-06-03.train-a.annotation.txt    joseph
    - train/2008-07-03.train-c.annotation.txt    jared
    - train/2008-10-02.train-c.annotation.txt    hussam
    - train/2009-01-05.train-a.annotation.txt    joseph
    - train/2009-03-25.train-a.annotation.txt    jonathan
    - train/2009-05-04.train-c.annotation.txt    jared
    - train/2009-05-08.train-c.annotation.txt    hussam
    - train/2009-07-02.train-c.annotation.txt    hussam
    - train/2009-11-13.train-c.annotation.txt    jared
    - train/2009-12-05.train-c.annotation.txt    hussam
    - train/2010-01-04.train-c.annotation.txt    jared
    - train/2010-02-13.train-a.annotation.txt    jonathan
    - train/2010-03-08.train-c.annotation.txt    jared
    - train/2010-03-20.train-c.annotation.txt    hussam
    - train/2010-04-12.train-c.annotation.txt    hussam
    - train/2010-05-30.train-c.annotation.txt    jared
    - train/2010-06-21.train-c.annotation.txt    hussam
    - train/2010-08-05.train-a.annotation.txt    vignesh
    - train/2010-08-15.train-c.annotation.txt    jared
    - train/2010-08-29.train-a.annotation.txt    vignesh
    - train/2010-10-17.train-c.annotation.txt    jared
    - train/2010-10-27.train-c.annotation.txt    hussam
    - train/2011-02-13.train-c.annotation.txt    hussam
    - train/2011-02-23.train-c.annotation.txt    hussam
    - train/2011-03-18.train-c.annotation.txt    jared
    - train/2011-04-14.train-c.annotation.txt    hussam
    - train/2011-04-17.train-c.annotation.txt    jared
    - train/2011-04-28.train-a.annotation.txt    jonathan
    - train/2011-08-17.train-a.annotation.txt    jonathan
    - train/2011-08-22.train-c.annotation.txt    jared
    - train/2011-11-24.train-c.annotation.txt    hussam
    - train/2011-12-07.train-c.annotation.txt    jared
    - train/2012-02-03.train-a.annotation.txt    joseph
    - train/2012-03-24.train-c.annotation.txt    jared
    - train/2012-05-04.train-a.annotation.txt    jonathan
    - train/2012-05-20.train-a.annotation.txt    vignesh
    - train/2012-06-02.train-a.annotation.txt    vignesh
    - train/2012-06-20.train-c.annotation.txt    hussam
    - train/2012-11-24.train-a.annotation.txt    jonathan
    - train/2012-11-30.train-a.annotation.txt    joseph
    - train/2012-12-15.train-a.annotation.txt    vignesh
    - train/2013-01-11.train-a.annotation.txt    jonathan
    - train/2013-01-30.train-a.annotation.txt    vignesh
    - train/2013-02-24.train-a.annotation.txt    vignesh
    - train/2013-05-05.train-a.annotation.txt    vignesh
    - train/2013-05-07.train-a.annotation.txt    vignesh
    - train/2013-05-19.train-a.annotation.txt    jonathan
    - train/2013-05-28.train-c.annotation.txt    jared
    - train/2013-07-10.train-a.annotation.txt    jonathan
    - train/2013-07-19.train-a.annotation.txt    jonathan
    - train/2013-08-29.train-c.annotation.txt    jared
    - train/2013-08-30.train-a.annotation.txt    jonathan
    - train/2013-09-12.train-a.annotation.txt    vignesh
    - train/2013-09-16.train-c.annotation.txt    jared
    - train/2013-10-04.train-a.annotation.txt    jonathan
    - train/2013-10-11.train-a.annotation.txt    jonathan
    - train/2013-10-28.train-a.annotation.txt    joseph
    - train/2013-12-02.train-a.annotation.txt    jonathan
    - train/2014-01-08.train-c.annotation.txt    jared
    - train/2014-08-14.train-c.annotation.txt    jared
    - train/2014-09-29.train-c.annotation.txt    jared
    - train/2014-12-21.train-c.annotation.txt    jared
    - train/2014-12-27.train-c.annotation.txt    jared
    - train/2015-01-20.train-b.annotation.txt    adjudicated
    - train/2015-02-04.train-b.annotation.txt    adjudicated
    - train/2015-04-19.train-b.annotation.txt    adjudicated
    - train/2015-05-08.train-c.annotation.txt    jared
    - train/2015-06-12.train-b.annotation.txt    adjudicated
    - train/2015-08-10.train-b.annotation.txt    adjudicated
    - train/2015-09-25.train-b.annotation.txt    adjudicated
    - train/2015-10-14.train-b.annotation.txt    adjudicated
    - train/2015-10-19.train-b.annotation.txt    adjudicated
    - train/2015-11-26.train-b.annotation.txt    adjudicated
    - train/2015-12-28.train-b.annotation.txt    adjudicated
    - train/2017-02-06.train-c.annotation.txt    jared
    - train/2017-03-02.train-c.annotation.txt    jared
    - train/2017-03-23.train-c.annotation.txt    jared
    - train/2017-05-09.train-c.annotation.txt    hussam
    - train/2017-07-15.train-c.annotation.txt    jared
    - train/2017-09-02.train-c.annotation.txt    hussam
    - train/2018-02-27.train-c.annotation.txt    jared
