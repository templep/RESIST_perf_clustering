import argparse
import pandas as pd
from os import listdir, makedirs
import os

import csv

# other python script including all functions to perform clusters over a performance matrix
import cluster
import preprocess_for_config as prep_config


###load one csv file specified by its path and filename
###the file is supposed to be a csv file that contains configurations of systems and performance measures over an execution
###return the number of line in the file and the data stored in the file
def load_csv(path, filename, with_name=False):
    filename = os.path.join(path, filename)
    data = pd.read_csv(filename, header=0)

    if with_name:
        data["inputname"] = os.path.splitext(os.path.basename(filename))[0]

    nb_data = data.shape[0]
    return data, nb_data


###find files in a specific directory with a specifc extension
###return a list of filename
def find_files(path, ext="csv"):
    # list all the files in the folder
    filenames = listdir(path)
    # list files that have the specified extension
    filename_list = [filename for filename in filenames if filename.endswith(ext)]
    return filename_list


### load and return all the files with a specific extension that are in a specific directory
### return the number of line in the file and a dataframe containing all data stored in the files that have been found
def load_all_csv(path, ext="csv", with_names=False):
    files_to_load = find_files(path, ext)
    ##print(len(files_to_load))
    ##print(files_to_load[0])
    ##print(files_to_load[1])
    # load first data file alone
    all_data, nb_config = load_csv(path, files_to_load[0], with_name=with_names)
    # load the rest and append to the previous dataframe
    for f in files_to_load[1:]:
        app_data, a = load_csv(path, f, with_name=with_names)
        all_data = pd.concat([all_data, app_data])
    # all_data = pd.concat([pd.read_csv(path+'/'+f) for f in files_to_load])

    return all_data, nb_config





### compute indexes of lines corresponding to each configurations
### function called to prepare an index list to be used by sort_data
def compute_index(all_data, nb_config):
    nb_rows = all_data.shape[0]
    idx = [i % nb_config for i in range(0, nb_rows)]
    # idx = idx%nb_config
    return idx


### save the two configs that form a U-shape in the dendrogram.
### idx1 and idx2 refer to indexes but it does not refer to indexes of configurations
### as idx1 and idx2 can be over the initial number of configurations.
### As we go up in the dendrogram, configurations are paired and this new pair is put at a new index.
### For instance, suppose indexes 1 and 2 are the closest and that we have a total of 10 initial configurations;
### indexes 1 and 2 will be merged together and a new index (i.e., 11) will be set. Next time, 11 will be merged with another index.
### Yet, in the file, index 1 will be stored instead of index 11 (which is abstract and does not refer to any configuration definition)
def save_config_clusters(output_dir, cfg_meas, idx1, idx2):
    # open a file to store the two configurations that contains the two indexes being merged and stored
    csvfile = open(
        output_dir + "comparison_" + str(idx1) + "_" + str(idx2) + ".csv",
        "w",
        newline="",
    )
    cfgwriter = csv.writer(
        csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )

    # write the header
    cfgwriter.writerow(cfg_meas.columns)

    # write the two configs and close. Again, note that they may refer to different indexes as indexes can refer to already merged configurations.
    cfgwriter.writerow(cfg_meas.iloc[idx1])
    cfgwriter.writerow(cfg_meas.iloc[idx2])
    csvfile.close()


### save two indexes that are merged in a dendrogram with the computed distance between the two.
def save_pairs_and_distance(
    output_dir, path, ext, children, distances, linkage="average", metric="cosine", is_on_input=False
):
    # create a dataframe with all needed informations that were given in parameteres
    pd_data = pd.DataFrame(data=(children[:, 0], children[:, 1], distances))

    # transpose the dataframe as it will be column-wise
    pd_data_save = pd_data.T

    # save the information in the desired file
    csvfile = (
        output_dir
        + "distances_and_pairs_link_"
        + str(linkage)
        + "_aff_"
        + str(metric)
        + "_level20_distance0_"
        + "on_input" if is_on_input else "on_config"
        +".csv"
    )
    pd_data_save.to_csv(csvfile)


def main(args):
    # store indexes of interest (retrieving performance values) to perform clustering and analyzes
    index_interest = args.idx_interest
    
    # will the clustering be done on the software configurations? If False, done on the input properties
    input_prop = args.is_on_input

    # load all data in a single dataframe
    path = args.folder
    ext = args.extension
    perf_matrix, nb_data = load_all_csv(path, ext)
    # print(perf_matrix.shape)

    # compute idexes for each configuration
    idx = compute_index(perf_matrix, nb_data)

    if(input_prop):
        # the lines of perf_matrix are already grouped by file
        # be aware that the loading of the files depend on the load_all_csv order
        data_per_cfg = perf_matrix
        
    else:
        # sort the lines of the dataframe by configuration
        data_per_cfg = prep_config.sort_data(perf_matrix, idx, nb_data)

    # remove configuration description to keep only measurements
    measures = cluster.extract_feature(data=data_per_cfg, nb_meas=args.nb_meas)
    # print(measures.shape)

    # create a dimension space in which each dimension corresponds to a measure observed from a test case
    feature_pts = cluster.create_feature_points(measures, nb_data, index_interest, input_prop)
    # print(feature_pts)
    # apply clustering and display dendrogram
    link = args.link
    metric = args.metric
    cls = cluster.cluster_to_display(
        feature_pts,
        n_clust=None,
        link=link,
        metric=metric,
        connect=None,
        cmpt_dist=True,
        threshold_dist=0,
    )

    ## save all pairs of configurations that form the dendrogram with the distances of their performance but in different files
    output_dir = os.path.join(
        args.output_folder,
        "comparison_pair_" + link + "_sim_" + metric + "_link_10_clusters/",
    )
    makedirs(output_dir, exist_ok=True)
    save_pairs_and_distance(
        output_dir, path, ext, cls.children_, cls.distances_, linkage=link, metric=metric, is_on_input=input_prop
    )

    ###################################################################
    ###################################################################
    ## try to understand how it works (the dendrogram + the hierarchical clustering); see how we can create a comparison score
    # cluster.retrieve_idx_per_cluster(cls)

    # print( str(len(cls.distances_)) + "   " + str(len(cls.children_)))
    # print(cls.children_)
    # print(cls.distances_)
    # print(cls.distances_[0])

    # cluster.compare_two_meas(feature_pts,0,44,index_interest)
    # cluster.compare_two_meas(feature_pts,66,55,index_interest)

    for i in range(len(cls.children_)):
        idx1 = cls.children_[i, 0]
        idx2 = cls.children_[i, 1]
        save_config_clusters(output_dir, perf_matrix, idx1, idx2)


#
# 	idx1 = cls.children_[1, 0]
# 	idx2 = cls.children_[1, 1]
#
# 	save_config_clusters(output_dir,perf_matrix,idx1,idx2)


### managing arguments
if __name__ == "__main__":
    # Define arguments for cli and run main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        help="The path to folder to find data to load",
        default="../data/res_ugc/",
        type=str,
    )
    parser.add_argument(
        "--extension",
        help="The extension file of files containing data",
        default="csv",
        type=str,
    )
    parser.add_argument(
        "--nb_meas",
        help="The number of performance measures per configuration on a single test case",
        default=8,
        type=int,
    )
    parser.add_argument(
        "--idx_interest",
        nargs="+",
        help="The indexes of the measures of performance to use to create clusters and analyze data. To use with multiple performance measures put the different indexes separated with spaces (e.g., to use with indexes 0 and 4 -> python3 main.py --idx_interest 0 4) all must be numerical",
        default=[0],
        type=int,
    )
    parser.add_argument(
        "--link",
        help="Defines the distance to use between clusters. Ultimately can help merging clusters together. See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html for possible values",
        default="average",
        type=str,
    )
    parser.add_argument(
        "--metric",
        help="Defines the metric to use when calculating distance between instances in a feature array.  See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html for possible values",
        default="cosine",
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder to compare two different configurations that are clustered together",
        default="../results/diff_config/",
        type=str,
    )
    parser.add_argument(
        "--is_on_input",
        help="if set to True, clustering will be done on input properties; if set to False, it will be done on software configurations",
        action='store_true',
    )
    args = parser.parse_args()
    
    main(args)
