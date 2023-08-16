#######################
### Import Packages ###
#######################

import numpy as np
import sys
import os
import scipy
import pandas as pd
from scipy.interpolate import interp1d

from bigbadbrain.utils import timing

@timing
def load_fictrac(directory, file='fictrac.dat'):
    """ Loads fictrac data from .dat file that fictrac outputs.

    Parameters
    ----------
    directory: string of full path to file
    file: string of file name

    Returns
    -------
    fictrac_data: pandas dataframe of all parameters saved by fictrac """

    for item in os.listdir(directory):
      if '.dat' in item:
        file = item

    with open(os.path.join(directory, file),'r') as f:
        df = pd.DataFrame(l.rstrip().split() for l in f)

        # Name columns
        df = df.rename(index=str, columns={0: 'frameCounter',
                                       1: 'dRotCamX',
                                       2: 'dRotCamY',
                                       3: 'dRotCamZ',
                                       4: 'dRotScore',
                                       5: 'dRotLabX',
                                       6: 'dRotLabY',
                                       7: 'dRotLabZ',
                                       8: 'AbsRotCamX',
                                       9: 'AbsRotCamY',
                                       10: 'AbsRotCamZ',
                                       11: 'AbsRotLabX',
                                       12: 'AbsRotLabY',
                                       13: 'AbsRotLabZ',
                                       14: 'positionX',
                                       15: 'positionY',
                                       16: 'heading',
                                       17: 'runningDir',
                                       18: 'speed',
                                       19: 'integratedX',
                                       20: 'integratedY',
                                       21: 'timeStamp',
                                       22: 'sequence'})

        # Remove commas
        for column in df.columns.values[:-1]:
            df[column] = [float(x[:-1]) for x in df[column]]

        fictrac_data = df
                
    # sanity check for extremely high speed (fictrac failure)
    speed = np.asarray(fictrac_data['speed'])
    max_speed = np.max(speed)
    if max_speed > 10:
        raise Exception('Fictrac ball tracking failed (reporting impossibly high speed).')
    return fictrac_data

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