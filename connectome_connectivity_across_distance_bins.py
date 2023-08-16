#######################
### Import Packages ###
#######################

import numpy as np
import os
import nibabel as nib
import ants
import brainsss
import matplotlib.pyplot as plt
import psutil
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.image import grid_to_graph
import time
import scipy
import itertools
import random
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid

#################
### FUNCTIONS ###
#################

def load_FDA():
    FDA_file = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/anat_templates/20220301_luke_2_jfrc_affine.nii'
    FDA = np.asarray(nib.load(FDA_file).get_fdata().squeeze(), dtype='float32')
    FDA = ants.from_numpy(FDA)
    FDA.set_spacing((.38,.38,.38))
    FDA_lowres = ants.resample_image(FDA,(2.6,2.6,5),use_voxels=False)
    return FDA, FDA_lowres

def load_synapses_in_FDA():
    synapses_in_FDA_file = '/oak/stanford/groups/trc/data/Yukun/syn_transformed_to_FDA.nii'
    synapses_in_FDA = np.asarray(nib.load(synapses_in_FDA_file).get_fdata().squeeze(), dtype='float32')
    return synapses_in_FDA

def get_hemibrain_bounding_box(synapses_in_FDA):
    # get axes edges in um space of hemibrain withing FDA space
    synapses_in_FDA = ants.from_numpy(synapses_in_FDA)
    synapses_in_FDA.set_spacing((.76,.76,.76))
    synapses_in_FDA_lowres = ants.resample_image(synapses_in_FDA,(2.6,2.6,5),use_voxels=False)
    synapses_in_FDA_lowres = synapses_in_FDA_lowres.numpy()
    
    start = {}
    stop = {}
    for axis, name in zip([0,1,2],['x','y','z']):
        
        start[name] = np.min(np.where(synapses_in_FDA_lowres != 0)[axis])
        start[name] = int(np.floor(start[name]))
        
        stop[name] = np.max(np.where(synapses_in_FDA_lowres != 0)[axis])
        stop[name] = int(np.ceil(stop[name]))
        
    return start, stop

######################
### Load synapses  ###
######################
#FDA is functional drosophila atlas

synapses_in_FDA = load_synapses_in_FDA()
synapses_in_FDA = ants.from_numpy(synapses_in_FDA)
synapses_in_FDA.set_spacing((.76,.76,.76))
synapses_in_FDA_lowres = ants.resample_image(synapses_in_FDA,(2.6,2.6,5),use_voxels=False)
synapses_in_FDA_lowres = synapses_in_FDA_lowres.numpy()

FDA, FDA_lowres = load_FDA()

start, stop = get_hemibrain_bounding_box(synapses_in_FDA.numpy())

FDA_cropped_to_hemi = FDA_lowres[start['x']:stop['x'],
                                 start['y']:stop['y'],
                                 start['z']:stop['z']]

dim_c = {'x': FDA_cropped_to_hemi.shape[0],
         'y': FDA_cropped_to_hemi.shape[1],
         'z': FDA_cropped_to_hemi.shape[2]}

##########################
### Load superclusters ###
##########################

n_clusters = 5000
clustering_dir = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210128_superv_simul_full_vol"
save_file = os.path.join(clustering_dir, '20220624_cluster_labels_flat.npy')
labels_flat = np.load(save_file)
labels_3d = np.reshape(labels_flat,(dim_c['x'],dim_c['y'],dim_c['z']))

cluster_sizes = []
for cluster in range(n_clusters):
    cluster_sizes.append(len(np.where(labels_flat==cluster)[0]))
background_cluster = np.argmax(cluster_sizes)

labels_3d = labels_3d.astype('float32')
labels_3d[np.where(labels_3d==background_cluster)] = np.nan

#now that we removed background cluster, lets give it to 0 so we can get rid of "0" as well
labels_3d[np.where(labels_3d==0)] = background_cluster
labels_flat = labels_3d.flatten()

#############################
### Load correlation maps ###
#############################

save_dir = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20220624_supervoxels_in_FDA'

save_file = os.path.join(save_dir, '20220624_ipsi_turn_corrs.npy')
ipsi_turn_corrs = np.load(save_file)

save_file = os.path.join(save_dir, '20220624_contra_turn_corrs.npy')
contra_turn_corrs = np.load(save_file)

save_file = os.path.join(save_dir, '20220624_fwd_corrs.npy')
fwd_corrs = np.load(save_file)

#############################################
### Load adjacency matrix of connectivity ###
#############################################

save_file = os.path.join(save_dir, 'adjacent_supervoxel.h5')

with h5py.File(save_file, 'r') as f:
    #print(list(f.keys()))
    adj = f['adjacent_supervoxel'][:]
    
# Feng removed the "0" entry so adding back
adj = np.pad(adj, (1,0), 'constant', constant_values=(0))

# Symmeterize
adj = (adj + adj.T) / 2

#############################################################
### DEFINE "YES" SUPERVOXELS and their pairwise distances ###
#############################################################
#yes meaning voxeles that are within the behavior correlation maps

threshold = .2
supervoxels_yes = list(np.where(ipsi_turn_corrs>threshold)[0])

centroids = []
for cluster_num in supervoxels_yes:
    xyz_voxel_space = np.asarray(np.where(labels_3d==cluster_num)).mean(axis=-1)
    centroids.append(list(xyz_voxel_space * (2.6,2.6,5)[0]))
centroids = np.asarray(centroids)

yes_distances = []
yes_combs = []
for comb in itertools.combinations(list(range(len(centroids))), 2):
    yes_combs.append(comb)
    id1 = comb[0]; id2 = comb[1]
    
    x_dist = (centroids[id1,0] - centroids[id2,0])**2
    y_dist = (centroids[id1,1] - centroids[id2,1])**2
    z_dist = (centroids[id1,2] - centroids[id2,2])**2
    dist = (x_dist + y_dist + z_dist)**.5
    yes_distances.append(dist)
yes_distances = np.asarray(yes_distances)

##############################
### CALCULATE CONNECTIVITY ###
##############################

lowers = np.arange(0,150,10)
uppers = np.arange(10,160,10)
sub_bin_width = 10

yes_connectivities = []
n_edges = []
for y in range(len(lowers)):
    labels_yes = list(np.where(((yes_distances > lowers[y]) & (yes_distances < uppers[y])))[0])
    n_edges.append(len(labels_yes))
    #labels_yes must first index into 

    yes_sample_connectivity = []
    for i in labels_yes:
        id1 = supervoxels_yes[yes_combs[i][0]]
        id2 = supervoxels_yes[yes_combs[i][1]]
        
        # correct for supervoxel size
        size = cluster_sizes[id1] + cluster_sizes[id2]
        yes_sample_connectivity.append(adj[id1,id2]/size)
    yes_connectivities.append(np.mean(yes_sample_connectivity))
# DONE! can plot yes_connectivities 

########################################################################
### REPEAT, BUT NOW WITH THE NULL DISTRIBUTION (CALCUALTE DISTANCES) ###
########################################################################

# get list of supervoxel ids that contain synapses
valid_clusters = list(np.where(np.sum(adj,axis=0) != 0)[0])
valid_clusters.remove(background_cluster)

### get distribution of distances between all supervoxels (VALID! supervoxels)

centroids = []
for cluster_num in valid_clusters:
    xyz_voxel_space = np.asarray(np.where(labels_3d==cluster_num)).mean(axis=-1)
    centroids.append(list(xyz_voxel_space * (2.6,2.6,5)[0]))
centroids = np.asarray(centroids)

combs = []
distances = []
for comb in itertools.combinations(list(range(len(centroids))), 2):
    combs.append(comb)
    id1 = comb[0]; id2 = comb[1]
    
    x_dist = (centroids[id1,0] - centroids[id2,0])**2
    y_dist = (centroids[id1,1] - centroids[id2,1])**2
    z_dist = (centroids[id1,2] - centroids[id2,2])**2
    dist = (x_dist + y_dist + z_dist)**.5
    distances.append(dist)
distances = np.asarray(distances)

###########################################################################
### REPEAT, BUT NOW WITH THE NULL DISTRIBUTION (CALCUALTE CONNECTIVITY) ###
###########################################################################

bootstrap_num = 1000
no_connectivities = []
for y in range(len(lowers)):
    
    supervoxels_no = ((distances > lowers[y]) & (distances < uppers[y]))

    # now I will want to take a specific number of these combos, randomly, n times
    labels_no = np.where(supervoxels_no)[0]
    sample_size = n_edges[y]
    
    no_connectivities_one_sample = []
    for n in range(bootstrap_num):
        one_sample_set = random.choices(labels_no,k=sample_size)
        sample_connectivity = []
        for i in range(sample_size):
            id1 = combs[one_sample_set[i]][0]
            id2 = combs[one_sample_set[i]][1]
            id1 = valid_clusters[id1]
            id2 = valid_clusters[id2]

            # correct for supervoxel size
            size = cluster_sizes[id1] + cluster_sizes[id2]
            
            sample_connectivity.append(adj[id1,id2]/size)

        no_connectivities_one_sample.append(np.mean(sample_connectivity))
    no_connectivities.append(np.asarray(no_connectivities_one_sample))
# DONE! can plot no_connectivities 