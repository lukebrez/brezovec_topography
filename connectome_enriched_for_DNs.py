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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
%matplotlib inline
import warnings
import networkx as nx
from fa2 import ForceAtlas2
from nxviz import CircosPlot
import nxviz as nv
import tqdm
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
import random
import time

connectome_dir = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20220624_supervoxels_in_FDA'

#############################
### load synapses and ids ###
#############################

file = os.path.join(connectome_dir,'hemibrain_all_neurons_synapses_polypre_centrifugal_synapses.pickle')
file = open(file, 'rb')
synapses = pickle.load(file)
cell_ids = np.unique(synapses['bodyid'])

#########################
### load cells in FDA ###
#########################

load_file = os.path.join(connectome_dir, 'synpervox.npy')
synpervox = np.load(load_file)

#########################
### connect to server ###
#########################

TOKEN = "" # <--- Paste your token here
c = Client('neuprint.janelia.org', 'hemibrain:v1.2.1', TOKEN)

########################
### load dice scores ###
########################

file = os.path.join(connectome_dir, 'all_neuron_dice.npy')
all_neuron_dice = np.load(file)
all_neuron_dice.shape

##############################
### FIND DNs in connectome ###
##############################

criteria = NC(type=['DNa.*', 'DNb.*', 'DNd.*', 'DNg.*', 'DNp.*', 'DNES.*', 'Giant_Fiber', 'MDN'])
neuron_df, roi_counts_df = fetch_neurons(criteria)
print(len(neuron_df['bodyId']))
DN_ids = list(neuron_df['bodyId'])
DN_names = list(neuron_df['instance'])

ids = []
names = []
for j,i in enumerate(DN_ids):
    try:
        ids.append(np.where(cell_ids==str(i))[0][0])
        names.append(DN_names[j])
    except:
        pass

#########################
### calculate DN DICE ###
#########################

mean_dn_dice = 0
dn_dices = []
for cell in ids:
    mean_dn_dice += all_neuron_dice[cell,beh]
    dn_dices.append(all_neuron_dice[cell,beh])
mean_dn_dice /= len(ids)

###################################
### BOOTSTRAP NULL DISTRIBUTION ###
###################################

num_bootstraps = 10000
sample_means = []
for sample in range(num_bootstraps):
    randomlist = random.sample(range(0, len(cell_ids)), len(ids))
    mean_dice = 0
    random_dices = []
    for cell in randomlist:
        mean_dice += all_neuron_dice[cell,beh]
    mean_dice /= len(ids)
    sample_means.append(mean_dice)

# DONE! COMPARE sample_means DISTRIUBUTION WITH mean_dn_dice
