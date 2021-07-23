# RESIST_perf_clustering
trying hierarchical clustering over performance matrix

This repo provides a first script to create a performance matrix of different variants of x264 executed over different videos.
Then, we apply hierarchical clustering and try to analyse generated clusters.

## Set-up
The scripts are written in Python3 (v3.8).
They use different libraries: numpy (written with v1.18) and pandas (written with v1.3) to manage structures and I/O operations over csv files.
They also use scipy (written with v1.7) and matplotlib (written with v3.4) to create and managing graphical plots (ie, dendograms and display of dendograms).
The cluster per se is done via scikit-learn (v0.24)
Please be sure that these librairies are installed.

## Folders
You will find two folders in this repo:
_data_ that contains all measures observed from different configurations of x264 when executed on different inputs.
The folder is structured per datasets. The subfolder _res\_ugc_ contains multiple files (one per video) in a csv format.
Inside each file (corresponding to a video), each line first describes the variants that have been used (one per line) and the last columns report the performance measures observed for different aspects.
8 different aspects have been observed from the size of the output video to the bitrate including the processing time.

_scripts_ gathers the different scripts that were developed to apply clustering on a dataset from the previous folder.
The scripts rely on the structure of the csv files and try to be as customizable as possible to try different parameter combinations from the clustering technique.

_results_ can be used to store dendograms from the different clustering configurations that were tried.
