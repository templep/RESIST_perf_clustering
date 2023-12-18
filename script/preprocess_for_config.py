#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:04:08 2023

@author: ptemple

This file contains the different functions that will preprocess data to prepare a dataframe nicely shaped for clustering over the software configurations.
The preprocessing functions regarding the clustering over input properties are placed in a different file.
Filtering files in the repo hierarchy and loading all the data (from csv files) are let in the main file.
"""

import pandas as pd


### deprecated use sort_data(data,idx,nb_config) instead
### grouping data from the same configuration so that they follow in the dataframe
def sort_data(data, nb_config):
    ids = data.iloc[data.iloc[:, 0] == data.iloc[0, 0]]
    sorted_data = ids
    for i in range(1, nb_config):
        ids = data.iloc[:, 0] == data.iloc[i, 0]
        sorted_data = pd.concat([sorted_data, ids])
    return sorted_data


### because all data have been loaded by file (ie, by test case), we need to group measures from the same configuration all together
### a configuration is a line (with associated measures) of the dataframe, the number of configuration does not change
### parameter idx has been computed a priori and helps to know which line corresponds to which configuration (supposed to speed up the process)
### return a new dataframe that have rearranged lines so that measures from the same configurations are in consecutive lines
def sort_data(data, idx, nb_config):
    # indexes of lines in data that correspond to configuration idx[0]
    ids = [a for a, v in enumerate(idx) if v == idx[0]]
    # store in the dataframe to be returned
    sorted_data = data.iloc[ids]
    # loop for all configurations
    for i in range(1, nb_config):
        ids = [a for a, v in enumerate(idx) if v == idx[i]]
        sorted_data = pd.concat([sorted_data, data.iloc[ids]])
    return sorted_data