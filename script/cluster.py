import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt


###return a dataframe containing only the last columns of an input dataframe
###the dataframe given in input is supposed to be built as first a description of the configuration as a vector (several columns of a single line)
###the remaining lines report performance observations of the configuration on a test case
###nb_meas is the number of performance measures that were done
###returns a dataframe containing only the performance measures, the index of configurations are not changed
def extract_feature(data,nb_meas=8):
	##compute index of columns of interest
	nb_col=data.shape[1]
	feat_config = nb_col-nb_meas
	idx_meas = range(feat_config,nb_col)
	measures = data.iloc[:,idx_meas]
	#print(measures.iloc[0,:])
	return measures

###this function creates virtually a high-dimension space in which a dimension is associated with a measure of the dataframe
###it prepares and structure the dataframe for clustering
###the input is composed of a dataframe containing different measures that will serve for clustering, the number of different configurations and the column(s) to keep
###it returns a new dataframe that contains a single line for each configuration, for each of them, the number of column is equal to the number of test cases and it reports the performance measure observed
###the returned dataframe is supposed to be homogeneous and contains only a number of lines equal to nb_config; the number of columns is only about the observed performance measure of a specific performance (such as execution time, size after compilation, etc.)
### is_on_config is a parameter to allow to perform clustering one way or the other (considering clustering by configurations or by input)
def create_feature_points(data, nb_config, meas_to_keep, is_on_config=True):
	nb_rows = data.shape[0]
	#nb_col = data.shape[1]
	#print(int(nb_rows/nb_config))
	
	data_extract = data.iloc[:,meas_to_keep]
	
	new_nb_r = nb_config
	new_nb_c = int(len(meas_to_keep)*(nb_rows/nb_config))
	if is_on_config:
		points = data_extract.to_numpy().reshape(new_nb_r,new_nb_c)
	else:
		points = data_extract.to_numpy().reshape(new_nb_c,new_nb_r)
	#print("shape dataset with only measures of interest")
	#print(points.shape)
	feature_points = pd.DataFrame(points)
	return feature_points
	
	
###function definition taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
###create a dendogram out of an agglomerative hierarchical clustering and displays it
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # compute linkage matrix to generate the dendrogram visualization
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    #### test to see the cluster associated with every leaf
    # print(d['color_list'])
    # print("####################################")
    # print(d['leaves_color_list'])
    # print("###################################")
	
###a function to perform agglomerative hierarchical clustering using scikit-learn defined functions
###it is configured to be displayable with the call to plot_dendogram
###data is the dataframe on which to apply the clustering algorithm
def cluster_to_display(data, n_clust=None, link='average', metric='cosine', connect=None, cmpt_dist=False,threshold_dist=0):

	model = AgglomerativeClustering(linkage=link, metric=metric, connectivity=connect, n_clusters=n_clust, compute_distances=cmpt_dist, distance_threshold=threshold_dist)
	m_to_plot=model.fit(data)
	print("HEEEEEEEEEEEEELLLLLLLLLLLLLLLLLLLLLLLOOOOOOOOOOOOOOOOOOOOOO")
	print(m_to_plot.n_clusters_)
	print(m_to_plot.labels_)
	print(m_to_plot.compute_distances)
	#print(m_to_plot.distances_)
	d = pd.DataFrame(m_to_plot.distances_)
	print("taille de la matrice distances_"+str(m_to_plot.distances_.shape))
	d.to_csv("../results/TEST_DISTANCES_on_input.csv")
	d = pd.DataFrame(m_to_plot.children_)
	d.to_csv("../results/TEST_CHILDREN_on_input.csv")
	
    #### plotting the dendrogam so that the colors reflect on the clusters' assignations is hard
    #### the only way I found to do it is to plot the labels on the leaves and deduce where should be the color_threshold
    #### NOW only the line with the fixed threshold is executing
	plt.title('Hierarchical Clustering Dendrogram')
    # to adapt the color_threshold value, please uncomment this line and comment the next one then visually find the right value and change the color threshold
	plot_dendrogram(m_to_plot, truncate_mode=None, p=200, labels=m_to_plot.labels_)
    #to adapt the color threshold, simply change the value and comment the previous line which defines the labels
	#plot_dendrogram(m_to_plot, truncate_mode='level', p=20, color_threshold=0.01)
	plt.xlabel("Number of points in node (or index of point if no parenthesis).")
	plt.show()
	
	return model

###a function to perform agglomerative hierarchical clustering using scikit-learn defined functions
###calls can be customized, not sure if it is displayable though
###data is the dataframe on which to apply the clustering algorithm
#def cluster(data, n_clusters=10, linkage='ward', affinity='euclidean', connectivity=None, compute_distances=False):
#	model = AgglomerativeClustering(linkage=linkage, affinity=affinity, connectivity=connectivity, n_clusters=n_clusters)
#	cls=model.fit(data)
#	return model



###################################################################
###################################################################
##tentative de comprehension pour automatisation de l'exploitation
	
def compare_two_meas(data, idx1, idx2,idx_meas):
	meas1 = data[idx1][idx_meas]
	meas2 = data[idx2][idx_meas]

	
	abs_diff = np.linalg.norm(meas1 - meas2)

def retrieve_idx_per_cluster(model):
	idx_max = model.labels_
	for i in range(model.n_clusters_):
		cfg = [a for a,v in enumerate(model.labels_) if v == i]
		print(str(i)+" : "+str(len(cfg))+" -> "+ str(cfg))

