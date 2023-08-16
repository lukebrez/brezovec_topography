### associated with figure 3

### Load temporal filters
filters = np.load(os.path.join(cluster_dir,'20230202_SC_temporal_filters.npy'))
# shape is (501, 4, 500), (supercluster, behavior_type, time)
# behavior types are 0:forward, 1:left_turn, 2:right_turn, 3:walking or not
# 250 is the center of the filter (instantaneous with behavior). Step is 20ms. (so 10sec width).

# three quality control thresholds
r_thresh = 0.5
derivative_thresh = 0.001
max_value = 0.1

break_points = []
mean_traces = []
flipped = []
for cluster in range(250):
    flip = False # this will handle the few filters that actually go negative 

    # the +250 gets the matching cluster in the other hemisphere
    a = filters[cluster,1,::-1]-filters[cluster+250,1,::-1]
    b = (filters[cluster,2,::-1]-filters[cluster+250,2,::-1])*-1

    r,_ = scipy.stats.pearsonr(a,b)
    if r < r_thresh:
        break_points.append(0)
        flipped.append(False)
        continue
    
    mean_trace = (a+b)/2
    
    ### see if should flip sign for the rare few
    extreme_v = mean_trace[np.argmax(np.abs(mean_trace))]
    if extreme_v < 0:
        a*=-1; b*=-1; mean_trace*=-1
        flip=True
        flipped.append(True)
    else:
        flipped.append(False)
    
    peak = np.argmax(mean_trace)
        
    if np.max(mean_trace) < max_value:
        break_points.append(0)
        continue
    
    # find where derivative exceeds thresh just before peak 
    break_div = np.where(np.diff(mean_trace)[:peak][::-1][20:] < derivative_thresh)[0][0] + 20
    break_point = peak - break_div
    break_points.append(break_point)