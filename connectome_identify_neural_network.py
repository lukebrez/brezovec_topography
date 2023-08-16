#######################
### IMPORT PACKAGES ###
#######################

from neuprint import Client
from neuprint import NeuronCriteria as NC
from neuprint import fetch_neurons
from neuprint import fetch_adjacencies
from neuprint.utils import connection_table_to_matrix
import bokeh.palettes
from bokeh.plotting import figure, show, output_notebook
output_notebook()
import hvplot.pandas
import holoviews as hv
import numpy as np
import pandas as pd
import ants
import nibabel as nib
import os
import pickle
import scipy
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
%matplotlib inline
import warnings
import networkx as nx
from fa2 import ForceAtlas2
from nxviz import CircosPlot
import nxviz as nv
import tqdm
import time
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy

def calc_dice(mask, neurons):
    intersect = np.logical_and(mask, neurons)
    dice = 2*np.sum(intersect)/(np.sum(neurons)+np.sum(mask))
    return dice

connectome_dir = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20220624_supervoxels_in_FDA'

#############################################
### load neuron-level connectivity matirx ###
#############################################

load_file = os.path.join(connectome_dir, '20220817_full_adj.npy')
full_adj = np.load(load_file)

#########################
### load cells in FDA ###
#########################

load_file = os.path.join(connectome_dir, 'synpervox.npy')
synpervox = np.load(load_file)
# this matrix describes for each neuron how many synapses are in each supervoxel

#####################
### load cell ids ###
#####################

file = os.path.join(connectome_dir,'hemibrain_all_neurons_synapses_polypre_centrifugal_synapses.pickle')
file = open(file, 'rb')
synapses = pickle.load(file)
cell_ids = np.unique(synapses['bodyid'])

############################
### load behavior scores ###
############################

unique_crop = np.load(os.path.join(connectome_dir, 'unique_glm_in_hemibrain.npy'))
#binarize
beh = 1
beh_mask = unique_crop[...,beh].copy()
beh_mask[np.where(beh_mask<.01)] = 0
beh_mask[np.where(beh_mask>=.01)] = 1

#################################################
### Mask the synapses using the behavior mask ###
#################################################
synpervox_mask = beh_mask[np.newaxis,...] * synpervox

###################################################################
### Calculate DICE score of every neuron with the behavior mask ###
###################################################################
synapse_thresh=1
synpervox_binary = np.where(synpervox>=synapse_thresh, 1, 0)
intersect = beh_mask[np.newaxis,...] * synpervox_binary
dices_OG = 2*np.sum(intersect,axis=(1,2,3))/(np.sum(synpervox_binary,axis=(1,2,3))+np.sum(beh_mask))

##################################################
### Calculate the FRAC_IN_MASK for each neuron ###
##################################################
synpervox_sum_all = np.sum(synpervox,axis=(1,2,3))
frac_in_mask = np.sum(synpervox_mask,axis=(1,2,3))/synpervox_sum_all

###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!###
#########################################
########### RUN GRID SEARCH #############
#########################################
###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!###

### DEFINE THRESHOLDS
frac_in_mask_thresholds = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]
dice_thresholds = np.linspace(0,.1,10)
conn_thresholds = [0,0.01,0.05,.1,.2,.3,.4,.5,.6,.8,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10,11,12,13,14,15]
synapse_thresholds = [1,5,10,20,30,40,50,60,70,80,90,100,150,200,300,400,500]

### two different initializations
initiation = 'frac_syn'

if initiation == 'dice':
    initiation_thresholds = dice_thresholds
    initiation_list = dices_OG
if initiation == 'frac_syn':
    initiation_thresholds = frac_in_mask_thresholds
    initiation_list = frac_in_mask

dice_outer_loop = []
num_cells_outer_loop = []
_cell_ids_outer = []
for initiation_threshold in initiation_thresholds:

    ##############################
    ### THRESHOLD FRAC_IN_MASK ###
    ##############################
    thresh_idx = np.searchsorted(np.sort(initiation_list), initiation_threshold)
    top_indicies = np.argsort(initiation_list)[thresh_idx:]
    top_cell_ids = cell_ids[top_indicies]
    top_cell_ids = [int(i) for i in top_cell_ids]
    FRAC_IN_MASK_PASS = top_cell_ids.copy()
    
    ####################
    ### MAKE SUB_ADJ ###
    ####################
    indicies = []
    for cell in FRAC_IN_MASK_PASS:
        try: # this is to ignore cells not in the original hemibrain dataset
            indicies.append(np.where(cell_ids_full_adj == cell)[0][0])
        except:
            pass
    syn_per_cell = np.sum(full_adj[indicies,:],axis=1) + np.sum(full_adj[:,indicies],axis=0)
    
    sub_adj = full_adj[indicies,:]
    sub_adj = sub_adj[:,indicies]
    sub_adj_ids = cell_ids_full_adj[indicies]
    
    ###############################################
    ### THRESHOLD SUB_ADJ BASED ON CONNECTIVITY ###
    ###############################################
    dice_inner_loop = []
    num_cells_inner_loop = []
    _cell_ids_inner = []
    for conn_thresh in conn_thresholds:
        mean_connectivity = np.asarray((np.mean(sub_adj/syn_per_cell,axis=0)) + np.asarray(np.mean(sub_adj/syn_per_cell,axis=1)))/2
        CONN_PASS = sub_adj_ids[np.where(mean_connectivity>conn_thresh)[0]]
        CONN_PASS = [int(i) for i in CONN_PASS]

        #####################
        ### CALULATE DICE ###
        #####################
        idx = []
        for cell in CONN_PASS:
            idx.append(np.where(cell_ids==str(cell))[0][0])
        cell_mask = np.sum(synpervox[idx,:,:,:],axis=0)

        ### Sweep synapse threshold ###
        dices = []
        for synapse_thresh in synapse_thresholds:
            cell_mask_binary = np.where(cell_mask>=synapse_thresh, 1, 0)
            dices.append(calc_dice(beh_mask[:75,:,:],cell_mask_binary[:75,:,:])) #cut at midline

        best_dice = np.max(dices)
        
        dice_inner_loop.append(best_dice)
        num_cells_inner_loop.append(len(CONN_PASS))
        _cell_ids_inner.append(CONN_PASS)
        
    dice_outer_loop.append(np.asarray(dice_inner_loop))
    num_cells_outer_loop.append(np.asarray(num_cells_inner_loop))
    _cell_ids_outer.append(np.asarray(_cell_ids_inner))

dice_outer_loop = np.asarray(dice_outer_loop) ### this is dice score of resulting networks
num_cells_outer_loop = np.asarray(num_cells_outer_loop) ### this is number of cells in the resulting networks

###################################################################
### MERGE NETWORKS BASED ON INFLECTION-POINT-DEFINED THRESHOLDS ###
###################################################################

a = np.where(num_cells_outer_loop<50)
aa=[]
for i in range(len(a[0])):
    aa.append(str(a[0][i])+str(a[1][i]))
    
b = np.where(dice_outer_loop>.25)
bb=[]
for i in range(len(b[0])):
    bb.append(str(b[0][i])+str(b[1][i]))
    
c=[]
for x in aa:
    if x in bb:
        c.append(x)
        
indicies = []
for x in c:
    indicies.append((int(x[0]),int(x[1:])))
    
cells = []
for idx in indicies:
    cells.extend(_cell_ids_outer[idx[0]][idx[1]])
cells = np.unique(cells)

### DONE!