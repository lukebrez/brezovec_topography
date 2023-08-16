#######################
### Import Packages ###
#######################

import os
import sys
import numpy as np
import argparse
import subprocess
import json
import time
from scipy.interpolate import interp1d
import scipy
import nibabel as nib
import bigbadbrain as bbb
import dataflow as flow

#####################
### Main Function ###
#####################

def main(args):

	### This fly class helps organize data for each fly
	class Fly:
		def __init__ (self, fly_name, fly_idx):
			self.dir = os.path.join(dataset_path, fly_name, 'func_0')
			self.fly_idx = fly_idx
			self.fly_name = fly_name
			self.maps = {}
		def load_timestamps (self):
			self.timestamps = bbb.load_timestamps(os.path.join(self.dir, 'imaging'))
		def load_fictrac (self):
			self.fictrac = Fictrac(self.dir, self.timestamps)
		def load_brain_slice (self):
			self.brain = brain[:,:,:,self.fly_idx]
		def load_anatomy (self):
			to_load = os.path.join(dataset_path, self.fly_name, 'warp', 'anat-to-meanbrain.nii')
			self.anatomy = np.array(nib.load(to_load).get_data(), copy=True)
		def load_z_depth_correction (self):
			to_load = os.path.join(dataset_path, self.fly_name, 'warp', '20201220_warped_z_depth.nii')
			self.z_correction = np.array(nib.load(to_load).get_data(), copy=True)
		def get_cluster_averages (self, cluster_model_labels, n_clusters):
			neural_data = self.brain.reshape(-1, 3384)
			signals = []
			self.cluster_indicies = []
			for cluster_num in range(n_clusters):
				cluster_indicies = np.where(cluster_model_labels==cluster_num)[0]
				mean_signal = np.mean(neural_data[cluster_indicies,:], axis=0)
				signals.append(mean_signal)
				self.cluster_indicies.append(cluster_indicies) # store for later
			self.cluster_signals=np.asarray(signals)
		def get_cluster_id (self, x, y):
			ax_vec = x*128 + y
			for i in range(n_clusters):
				if ax_vec in self.cluster_indicies[i]:
					cluster_id = i
					break
			return cluster_id

	### This fictrac class helps process behavior data
	class Fictrac:
		def __init__ (self, fly_dir, timestamps):
			self.fictrac_raw = bbb.load_fictrac(os.path.join(fly_dir, 'fictrac'))
			self.timestamps = timestamps
		def make_interp_object(self, behavior):
			# Create camera timepoints
			fps=50
			camera_rate = 1/fps * 1000 # camera frame rate in ms
			expt_len = 1000*30*60
			x_original = np.arange(0,expt_len,camera_rate)

			# Smooth raw fictrac data
			fictrac_smoothed = scipy.signal.savgol_filter(np.asarray(self.fictrac_raw[behavior]),25,3)

			# Create interp object with camera timepoints
			fictrac_interp_object = interp1d(x_original, fictrac_smoothed, bounds_error = False)
			return fictrac_smoothed, fictrac_interp_object

		def pull_from_interp_object(self, interp_object, timepoints):
			new_interp = interp_object(timepoints)
			np.nan_to_num(new_interp, copy=False);
			return new_interp

		def interp_fictrac(self):
			behaviors = ['dRotLabY', 'dRotLabZ']; shorts = ['Y', 'Z']
			self.fictrac = {}

			for behavior, short in zip(behaviors, shorts):
				raw_smoothed, interp_object = self.make_interp_object(behavior)
				self.fictrac[short + 'i'] = interp_object
				self.fictrac[short] = raw_smoothed

		def make_walking_vector(self):
			self.fictrac['W'] = np.zeros(len(self.fictrac['Y']))
			YZ = np.sqrt(np.power(self.fictrac['Y']/np.std(self.fictrac['Y']),2),
				 np.power(self.fictrac['Z']/np.std(self.fictrac['Z']),2))
			self.fictrac['W'][np.where(YZ>.2)] = 1

			fps=50
			camera_rate = 1/fps * 1000 # camera frame rate in ms
			expt_len = 1000*30*60
			x_original = np.arange(0,expt_len,camera_rate)
			self.fictrac['Wi'] = interp1d(x_original, self.fictrac['W'], bounds_error = False, kind = 'nearest')

	### this function loops over every time-step in time_shifts to build a full matrix of each time-stamp
	### this is done for a given fly, and given z-place, and a given behavior
	def build_timeshifted_behavior_matrix(time_shifts, fly, z, behavior):
		# Get correct behavior interp obj
		if 'Z' in behavior: behavior_i = 'Zi'
		if 'Y' in behavior: behavior_i = 'Yi'

		interp_obj = flies[fly].fictrac.fictrac[behavior_i]

		behavior_shifts = []
		for shift in time_shifts:
			fictrac_interp = interp_obj(flies[fly].timestamps[:,z]+shift)
			fictrac_interp = np.nan_to_num(fictrac_interp)

			# Split VELOCITY in +/-
			if 'pos' in behavior:
				fictrac_interp = np.clip(fictrac_interp, a_min=0, a_max=None)
			if 'neg' in behavior:
				fictrac_interp = np.clip(fictrac_interp, a_min=None, a_max=0)*-1

			behavior_shifts.append(fictrac_interp)

		return time_shifts, behavior_shifts

	### This function will create a matrix that, for every neural time point, saves a window of behavior
	### interpolated to match the given vector of time_shifts
	def build_X (time_shifts, behaviors, z):
		all_fly_shifts = []
		for fly in fly_names:
			all_behavior_shifts = []
			for behavior in behaviors:
				time_shifts, behavior_shifts = build_timeshifted_behavior_matrix(time_shifts=time_shifts,
																				 fly=fly,
																				 z=z,
																				 behavior=behavior)
				all_behavior_shifts.append(np.asarray(behavior_shifts))
			all_behavior_shifts = np.asarray(all_behavior_shifts)
			all_behavior_shifts = np.reshape(all_behavior_shifts, (-1,3384))
			all_fly_shifts.append(all_behavior_shifts)
		X = np.asarray(all_fly_shifts)
		return X

	logfile = args['logfile']
	printlog = getattr(flow.Printlog(logfile=logfile), 'print_to_log')
	fly_names = ['fly_087', 'fly_089', 'fly_094', 'fly_097', 'fly_098', 'fly_099', 'fly_100', 'fly_101', 'fly_105']
	dataset_path = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20190101_walking_dataset"
	expt_len = 1000*30*60
	resolution = 10
	high_res_timepoints = np.arange(0,expt_len,resolution) #0 to last time at subsample res

	###################
	### Build Flies ###
	###################
	### loop  over flies and load and process neural and behavior data based on classes defined above
	flies = {}
	for i, fly in enumerate(fly_names):
		flies[fly] = Fly(fly_name=fly, fly_idx=i)
		flies[fly].load_timestamps()
		flies[fly].load_fictrac()
		flies[fly].fictrac.interp_fictrac()
		flies[fly].fictrac.make_walking_vector()

	### these timeshifts define what temporal points to interpolate behavior at relative to 
	### neural activity at t=0
	time_shifts = list(range(-5000,5000,20)) # in ms
	behaviors = ['Y_pos_plus', 'Z_pos_plus', 'Z_neg_plus',]

	############################################
	### Build the complete X behavior matrix ###
	############################################
	Xs = []
	### Loop over z-slices
	for z in range(49):
		printlog(str(z))
		X = build_X(time_shifts, behaviors, z) # build the X behavior matrix
		Xs.append(X)

	######################
	### Save Responses ###
	######################
	save_file = F"/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210316_neural_weighted_behavior/master_X"
	np.save(save_file, np.asarray(Xs))

	##########################################
	### Weight behavior by neural activity ###
	##########################################
	### Now that we have created the X behavior matrix, we can weigh each time window by neural activity
	for z in range(9,49-9):
			printlog(f"Z:{z}")

			#######################
			### Load Superslice ###
			#######################
			### A superslice is a single z-plane but all flies have already been concatenated along an axis of this array
			brain_file = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20201129_super_slices/superslice_{}.nii".format(z)
			brain = np.array(nib.load(brain_file).get_data(), copy=True)
			# Delete a fly that is in the superslice but was excluded from all analysis due to not passing quality control
			fly_idx_delete = 3 #(fly_095)
			brain = np.delete(brain, fly_idx_delete, axis=-1) #### DELETING FLY_095 ####

			# Get cluster responses for this slice
			for fly in fly_names:
				flies[fly].load_brain_slice()
				flies[fly].get_cluster_averages(cluster_model_labels_all[z,:], n_clusters)

			#################
			### Main Loop ###
			#################
			cluster_responses = []
			for cluster_num in range(n_clusters):
			    if cluster_num%100 == 0:
			        printlog(str(cluster_num))
			    ###############################################################
			    ### Build Y vector for a single supervoxel (with all flies) ###
			    ###############################################################
			    all_fly_neural = []
			    for fly in fly_names:
			        signal = flies[fly].cluster_signals[cluster_num,:]
			        all_fly_neural.extend(signal)
			    Y = np.asarray(all_fly_neural)

			    ###########################################
			    ### Build the X matrix for this cluster ###
			    ###########################################
			    # For each fly, this cluster could have originally come from a different z-depth
			    # Get correct original z-depth
			    Xs_new = []
			    for i, fly in enumerate(fly_names):
			        cluster_indicies = flies[fly].cluster_indicies[cluster_num]
			        z_map = flies[fly].z_correction[:,:,z].ravel()
			        original_z = int(np.median(z_map[cluster_indicies]))
			        Xs_new.append(X[original_z,i,:,:])
			    Xs_new = np.asarray(Xs_new)
			    X_cluster = np.reshape(np.moveaxis(Xs_new,0,1),(-1,30456))

			    ###################
			    ### Dot Product ###
			    ###################
			    ### this is where the magic happens
			    cluster_response = np.dot(X_cluster,Y)
			    cluster_responses.append(cluster_response)
			cluster_responses = np.asarray(cluster_responses)

			######################
			### Save Responses ###
			######################
			save_file = F"/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20210316_neural_weighted_behavior/responses_{z}"
			np.save(save_file, cluster_responses)
			brain = None
			Y = None

if __name__ == '__main__':
	main(json.loads(sys.argv[1]))	 