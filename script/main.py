import argparse
import pandas as pd
from os import listdir

import csv

#other python script including all functions to perform clusters over a performance matrix
import cluster


###load one csv file specified by its path and filename
###the file is supposed to be a csv file that contains configurations of systems and performance measures over an execution
###return the number of line in the file and the data stored in the file
def load_csv(path, filename):
	filename = path+filename
	data = pd.read_csv(filename)
	nb_data = data.shape[0]
	return data, nb_data

###find files in a specific directory with a specifc extension
###return a list of filename
def find_files (path, ext="csv"):
	ext = ext
	#list all the files in the folder
	filenames = listdir(path)
	#list files that have the specified extension
	filename_list = [filename for filename in filenames if filename.endswith(ext)]
	return filename_list

### load and return all the files with a specific extension that are in a specific directory
### return the number of line in the file and a dataframe containing all data stored in the files that have been found
def load_all_csv(path, ext="csv"):
	files_to_load = find_files(path,ext)
	##print(len(files_to_load))
	##print(files_to_load[0])
	##print(files_to_load[1])
	#load first data file alone
	all_data,nb_config = load_csv(path,files_to_load[0])
	#load the rest and append to the previous dataframe
	for f in files_to_load[1:]:
		app_data, a = load_csv(path,f)
		all_data = pd.concat([all_data,app_data])
#	all_data = pd.concat([pd.read_csv(path+'/'+f) for f in files_to_load])
			
	return all_data,nb_config

### deprecated use sort_data(data,idx,nb_config) instead
### grouping data from the same configuration so that they follow in the dataframe
def sort_data(data, nb_config):
	ids = data.iloc[data.iloc[:,0] == data.iloc[0,0]]
	sorted_data = ids
	for i in range(1,nb_config):
		ids = data.iloc[:,0] == data.iloc[i,0]
		sorted_data = pd.concat([sorted_data, ids])
	return sorted_data

### because all data have been loaded by file (ie, by test case), we need to group measures from the same configuration all together
### a configuration is a line (with associated measures) of the dataframe, the number of configuration does not change
### parameter idx has been computed a priori and helps to know which line corresponds to which configuration (supposed to speed up the process)
### return a new dataframe that have rearranged lines so that measures from the same configurations are in consecutive lines
def sort_data(data, idx, nb_config):
	#indexes of lines in data that correspond to configuration idx[0]
	ids = [a for a,v in enumerate(idx) if v == idx[0]]
	#store in the dataframe to be returned
	sorted_data = data.iloc[ids]
	#loop for all configurations
	for i in range(1,nb_config):
		ids = [a for a,v in enumerate(idx) if v == idx[i]]
		sorted_data = pd.concat([sorted_data, data.iloc[ids]])
	return sorted_data

### compute indexes of lines corresponding to each configurations
### function called to prepare an index list to be used by sort_data
def compute_index(all_data, nb_config):
	nb_rows = all_data.shape[0]
	idx = [i%nb_config for i in range(0,nb_rows)]
	#idx = idx%nb_config
	return idx
	
def save_config_clusters(output_dir,cfg_meas,idx1,idx2):
	csvfile = open(output_dir+"comparison_"+str(idx1)+"_"+str(idx2)+".csv",'w',newline='')
	cfgwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	cfgwriter.writerow(cfg_meas.iloc[idx1])
	cfgwriter.writerow(cfg_meas.iloc[idx2])
	csvfile.close()
	

def main(args):

	#store indexes of interest (retrieving performance values) to perform clustering and analyzes
	index_interest = args.idx_interest

	#load all data in a single dataframe
	path = args.folder
	ext = args.extension
	perf_matrix,nb_data = load_all_csv(path,ext)
	#print(perf_matrix.shape)
	
	#compute idexes for each configuration
	idx = compute_index(perf_matrix,nb_data)
	
	#sort the lines of the dataframe by configuration
	data_per_cfg = sort_data(perf_matrix, idx, nb_data)
	##tests
	#print(data_per_cfg.shape)
	#data_per_cfg = sort_data(perf_matrix,nb_data)
	#print(data_per_cfg.shape)
	#print(data_per_cfg[0])
	#print(data_per_cfg[1])
	
	#remove configuration description to keep only measurements
	measures = cluster.extract_feature(data=data_per_cfg,nb_meas=args.nb_meas)
	#print(measures.shape)
	
	#create a dimension space in which each dimension corresponds to a measure observed from a test case
	feature_pts = cluster.create_feature_points(measures, nb_data, index_interest)
	#print(feature_pts)
	#apply clustering and disply dendogram
	cls = cluster.cluster_to_display(feature_pts)
	#cluster.cluster(feature_pts)
	
	
	
	###################################################################
	###################################################################
	##tentative de comprehension pour automatisation de l'exploitation
	cluster.retrieve_idx_per_cluster(cls)
		
	#print( str(len(cls.distances_)) + "   " + str(len(cls.children_)))
	#print(cls.children_)
	#print(cls.distances_)
	#print(cls.distances_[0])
	
	#cluster.compare_two_meas(feature_pts,0,44,index_interest)
	#cluster.compare_two_meas(feature_pts,66,55,index_interest)
	
	
	output_dir = args.output_folder
	idx1 = cls.children_[0, 0]
	idx2 = cls.children_[0, 1]
	
	save_config_clusters(output_dir,perf_matrix,idx1,idx2)
	
	idx1 = cls.children_[1, 0]
	idx2 = cls.children_[1, 1]
	
	save_config_clusters(output_dir,perf_matrix,idx1,idx2)
	
	
### managing arguments
if __name__ == '__main__':
	# Define arguments for cli and run main function
	parser = argparse.ArgumentParser()
	parser.add_argument('--folder', help="The path to folder to find data to load",default="../data/res_ugc/",type=str)
	parser.add_argument('--extension', help="The extension file of files containing data",default="csv",type=str)
	parser.add_argument('--nb_meas', help="The number of performance measures per configuration on a single test case",default=8,type=int)
	parser.add_argument('--idx_interest', nargs='+', help="The indexes of the measures of performance to use to create clusters and analyze data. To use with multiple performance measures put the different indexes separated with spaces (e.g., to use with indexes 0 and 4 -> python3 main.py --idx_interest 0 4) all must be numerical",default=0,type=float)
	parser.add_argument('--output_folder', help="Output folder to compare two different configurations that are clustered together", default="../results/diff_config/",type=str)
	args = parser.parse_args()
	main(args)


