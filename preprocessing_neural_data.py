#########################
### Motion Correction ###
#########################

def make_temporal_mean(brain):
	meanbrain = np.mean(brain, axis=-1)
	return meanbrain

def motion_correct_single_volume(meanbrain, single_red_vol):
	motCorr_vol = ants.registration(meanbrain, single_red_vol, type_of_transform='SyN')
	return motCorr_vol

def apply_transforms_from_red_to_green_channel(meanbrain, motCorr_vol, single_green_vol):
	transformlist = motCorr_vol['fwdtransforms']
	moco_green = ants.apply_transforms(meanbrain, single_green_vol, transformlist)
	return moco_green

#######################################################
### High-pass filter each voxel's temporal activity ###
#######################################################

def high_pass_filter(brain):
	smoothed = gaussian_filter1d(brain,sigma=200,axis=-1,truncate=1)
	brain_corrected = brain - smoothed + np.mean(brain, axis=3)[:,:,:,None] #need to add back in mean to preserve offset
	return brain_corrected

##############################################
### Z-score each voxel's temporal activity ###
##############################################

def z_score(brain):
	brain_mean  = np.mean(brain, axis=3)
	brain_std = np.std(brain, axis=3)
	brain = (brain - brain_mean[:,:,:,None]) / brain_std[:,:,:,None]
	return brain