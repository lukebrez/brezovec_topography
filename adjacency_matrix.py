#######################
### IMPORT PACKAGES ###
#######################

from collections import defaultdict
from collections import Counter
import numpy as np
from tqdm import tqdm_notebook,tqdm
import copy
import os
import pickle
from multiprocessing import Pool
import multiprocessing
from functools import partial
from multiprocessing import Semaphore
import time
import tables
import h5py

########################
### DEFINE FUNCTIONS ###
########################

class connectome:
    def __init__(self,load_data=True,data_dir=''):
        self.synapse_data=None
        self.neuron_data=None
        self.grids=None
        self.voxel_size=None
        if load_data:
            with open(os.path.join(data_dir,'hemibrain_all_neurons_synapses_polypre_centrifugal_synapses.pickle'), 'rb') as handle:
                self.synapse_data=pickle.load(handle) 
            with open(os.path.join(data_dir,'hemibrain_all_neurons_metrics_polypre_centrifugal_synapses.pickle'), 'rb') as handle:
                self.neuron_data=pickle.load(handle)
            with open(os.path.join(data_dir,'hemibrain_all_neurons_metrics_polypre_centrifugal_distance.pickle'), 'rb') as handle:
                self.neuron_data2=pickle.load(handle)
            for i in ['x','y','z']:
                self.synapse_data[i]=SEM_voxel_size*np.array(self.synapse_data[i]).astype('float')
    @staticmethod
    def multiprocessing_help_func_2(x_set,y_set,z_set,i,return_dict):
        tempt_list_y=[]
        for _,j in enumerate(y_set):
            tempt_list_z=[]
            for k in z_set:
                tempt_list_z.append(x_set[i].intersection(j).intersection(k))
            tempt_list_y.append(tempt_list_z)
        return_dict[i]=tempt_list_y
        return i,tempt_list_y
    def create_set_list_1d(self,data,bins,cpu=96):
        def multiprocessing_help_func_1(data,bins,i):
            return set(list(np.where((data>=bins[i])*(data<=bins[i+1]))[0]))
        set_list=[]
        for i in range(len(bins)-1):
            set_list.append(multiprocessing_help_func_1(data,bins,i))
        return set_list
    def create_grid(self,x_0,y_0,z_0):
        self.voxel_size=(x_0,y_0,z_0)
        x_ = np.arange(self.synapse_data['x'].min()//x_0*x_0,self.synapse_data['x'].max()//x_0*x_0+2*x_0,x_0)
        y_ = np.arange(self.synapse_data['y'].min()//y_0*y_0,self.synapse_data['y'].max()//y_0*y_0+2*y_0,y_0)
        z_ = np.arange(self.synapse_data['z'].min()//z_0*z_0,self.synapse_data['z'].max()//z_0*z_0+2*z_0,z_0)
        self.grids=[x_,y_,z_]
    def grid_data(self):
        x_set=self.create_set_list_1d(self.synapse_data['x'],self.grids[0])
        y_set=self.create_set_list_1d(self.synapse_data['y'],self.grids[1])
        z_set=self.create_set_list_1d(self.synapse_data['z'],self.grids[2])
        print('Done grid')
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs=[]
        out_list = list()
        rets=[]
        for i in range(0, len(x_set)):
            process = multiprocessing.Process(target=self.multiprocessing_help_func_2, 
                                              args=(x_set,y_set,z_set,i,return_dict))
            jobs.append(process)
        # Start the processes
        for j in jobs:
            j.start()
        # Ensure all of the processes have finished
        for j in jobs:
            j.join()
        print('Finished griding, combining all results')
        grid_list=[]
        for i in range(0,len(x_set)):
            grid_list.append(return_dict[i])
        grid_list=np.array(grid_list)
        self.grid_list=np.array(grid_list)
        return grid_list

class adj_cal(connectome):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.bodyids=set(list(self.synapse_data['bodyid'].astype(int))+list(self.synapse_data['partner'].astype(int)))
        
    def processing_voxel(self):
        #split input and output
        n1,n2,n3=self.grid_list.shape
        self.n1,self.n2,self.ne=n1,n2,n3
        return_list=np.empty( (n1,n2,n3), dtype=object)
        for i in tqdm(range(n1)):
            for j in range(n2):
                for k in range(n3):
                    tempt_list=list(self.grid_list[i][j][k])
                    inputs_from_neruons_known_grid_list=[]
                    outputs_to_neruons_grid_list=[]
                    for l in tempt_list:
                        if int(self.synapse_data['prepost'][l])==0:
                            inputs_from_neruons_known_grid_list.append(int(self.synapse_data['bodyid'][l]))
                            outputs_to_neruons_grid_list.append(int(self.synapse_data['partner'][l]))
                        elif int(self.synapse_data['prepost'][l])==1:
                            inputs_from_neruons_known_grid_list.append(int(self.synapse_data['partner'][l]))
                            outputs_to_neruons_grid_list.append(int(self.synapse_data['bodyid'][l])) 
                    return_list[i,j,k]=np.stack([inputs_from_neruons_known_grid_list,outputs_to_neruons_grid_list])
        self.grid_list_inoutput=return_list
        return return_list
    def preprocessing_for_adj(self):
        self.synapse_data['input']=[]
        self.synapse_data['output']=[]
        for l in tqdm(range(len(self.synapse_data['bodyid']))):
            if int(self.synapse_data['prepost'][l])==0:
                self.synapse_data['input'].append(int(self.synapse_data['bodyid'][l]))
                self.synapse_data['output'].append(int(self.synapse_data['partner'][l]))
            else:
                self.synapse_data['output'].append(int(self.synapse_data['bodyid'][l]))
                self.synapse_data['input'].append(int(self.synapse_data['partner'][l]))
        self.synapse_data['input']=np.array(self.synapse_data['input'])
        self.synapse_data['output']=np.array(self.synapse_data['output'])
        self.neuron_to_synapse_input=defaultdict(list)
        self.neruon_to_synapse_output=defaultdict(list)
        for l in tqdm(range(len(self.synapse_data['bodyid']))):
            self.neuron_to_synapse_input[self.synapse_data['input'][l]].append(l) #synapse inputs to neurons
            self.neruon_to_synapse_output[self.synapse_data['output'][l]].append(l)
        x0,y0,z0 = self.voxel_size
        self.synapse_data['x_grid']=((self.synapse_data['x']-self.grids[0][0])//x0).astype('int')
        self.synapse_data['y_grid']=((self.synapse_data['y']-self.grids[1][0])//y0).astype('int')
        self.synapse_data['z_grid']=((self.synapse_data['z']-self.grids[2][0])//z0).astype('int')
        
    def calculate_adj(self,cpu=96,name='adjacent'):
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        for i in np.arange(self.grid_list_inoutput.ravel().shape[0]):
            input_queue.put(i)
        return_queue = manager.Queue()

        jobs=[]
        out_list = list()
        rets=[]
        proc = multiprocessing.Process(target=self.handle_output, args=(name+'.h5',
                                    name,(self.grid_list_inoutput.ravel().shape[0], self.grid_list_inoutput.shape[0], self.grid_list_inoutput.shape[1], self.grid_list_inoutput.shape[2]), return_queue, ))
        proc.start()


        for i in range(cpu-1):
            process = multiprocessing.Process(target=calculate_cor_multiprocessing, 
                                              args=(self.grid_list_inoutput.ravel(), input_queue, return_queue, None, self.synapse_data, self.neuron_to_synapse_input, self.grid_list_inoutput.shape ))
            jobs.append(process)
            process.start()
            
        for j in jobs:
            j.join()

        return_queue.put(None)
        proc.join()

    def handle_output(self,name,d_name,size,return_queue):
        assert not os.path.exists(name)
        hdf = h5py.File(name, mode='w')
        hdf.create_dataset('grids_x', data=self.grids[0])
        hdf.create_dataset('grids_y', data=self.grids[1])
        hdf.create_dataset('grids_z', data=self.grids[2])
        hdf.create_dataset('voxel_size', data=np.array(self.voxel_size))
        dset = hdf.create_dataset(d_name, size, dtype='float')
        dset = hdf[d_name]
        while True:
            args = return_queue.get()
            if args:
                dset[args[0]]=args[1]
            else:
                break
        hdf.close()

def calculate_cor_multiprocessing(grid_list,input_queue, return_queue, sema, synapse_data, neuron_to_synapse_input, n_shape):
        while not input_queue.empty():
            probe1 = input_queue.get()
            place_holder=np.zeros(n_shape)
            n_out_list=grid_list[probe1][1] #probe1 outputs to neurons n_out_list
            for bid in n_out_list:
                for l2 in neuron_to_synapse_input[bid]:
                    j1,j2,j3=synapse_data['x_grid'][l2],synapse_data['y_grid'][l2],synapse_data['z_grid'][l2] #inputs to neuron
            return_queue.put((probe1,place_holder))

def calculate_adj(cpu=96,name='adjacent_supervoxel'):
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    for i in np.arange(1,n_voxel):
        input_queue.put(i)
    return_queue = manager.Queue()
    jobs=[]
    out_list = list()
    rets=[]   
    proc = multiprocessing.Process(target=handle_output, args=(name+'.h5',
                                name,(n_voxel,n_voxel), return_queue, ))
    proc.start()
    for i in range(cpu-1):
        process = multiprocessing.Process(target=calculate_help_multiprocessing, 
                                          args=(c.grid_list_inoutput, input_queue, return_queue, None, 
                                                c.synapse_data, c.neuron_to_synapse_input, n_voxel, supervoxel_to_voxel,voxel_to_supervoxel))
        jobs.append(process)
        process.start()

    for j in jobs:
        j.join()

    return_queue.put(None)
    proc.join()

def handle_output(name,d_name,size,return_queue):
    assert not os.path.exists(name)
    hdf = h5py.File(name, mode='w')
    dset = hdf.create_dataset(d_name, size, dtype='float')
    dset = hdf[d_name]
    while True:
        args = return_queue.get()
        if args:
            dset[args[0]]=args[1]
        else:
            break
    hdf.close()

def calculate_help_multiprocessing(grid_list,input_queue, return_queue, sema, synapse_data, neuron_to_synapse_input, n_shape, supervoxel_to_voxel,voxel_to_supervoxel):
        while not input_queue.empty():
            probe1 = input_queue.get()
            place_holder=np.zeros(n_shape)
            voxel_indices=supervoxel_to_voxel[probe1].T
            n_out_list=[]
            for jjj in range(len(voxel_indices)):
                if voxel_indices[jjj][0]<x_offset or voxel_indices[jjj][1]<y_offset or voxel_indices[jjj][2]<z_offset:
                    continue
                if voxel_indices[jjj][0]-x_offset>=700 or voxel_indices[jjj][1]-y_offset>=591 or voxel_indices[jjj][2]-z_offset>=392:
                    continue
                n_out_list.extend(grid_list[voxel_indices[jjj][0]-x_offset,voxel_indices[jjj][1]-y_offset,voxel_indices[jjj][2]-z_offset][1])
            for bid in n_out_list:
                for l2 in neuron_to_synapse_input[bid]:
                    j1,j2,j3=synapse_data['x_grid'][l2],synapse_data['y_grid'][l2],synapse_data['z_grid'][l2] #inputs to neuron
                    if not np.isnan(voxel_to_supervoxel[j1,j2,j3]):
                        place_holder[int(voxel_to_supervoxel[j1,j2,j3])-1]+=1
            return_queue.put((probe1,place_holder))

#####################
### Calculate Adj ###
#####################

SEM_voxel_size=0.38
n_voxel=4999

x_offset=312
y_offset=30
z_offset=58

voxel_to_supervoxel=np.load('/home/data/supervoxels/20220701_supervoxel_labels_in_fda.npy')
voxel_to_supervoxel=voxel_to_supervoxel[x_offset:][:,y_offset:][:,:,z_offset:]

c=adj_cal()
c.create_grid(0.38,0.38,0.38)
c.grid_data();
c.processing_voxel()
c.preprocessing_for_adj()

with open('supervoxel_to_voxel.pickle', 'rb') as handle:
    supervoxel_to_voxel = pickle.load(handle)

calculate_adj(cpu=96,name='adjacent_supervoxel')

### DONE!