import numpy as np
import metric
from saliency_metric.benchmark.metrics import *




def validation(solver_instance, dataset_instance):
	loss_list = []
	cc_list = []
	sim_list = []
	while dataset_instance.finished_epoch < 1:
	    frame_minibatch, _ = tranining_dataset.next_hdr_batch(1,stops=args.stops)

	    solver.net.blobs['data'].data[...] = frame_minibatch
	    # solver.net.blobs['gt'].data[...] = density_minibatch
	    solver.net.forward()
	    loss = solver.net.blobs['loss'].data[...]
	    prediction = solver.net.blobs['predict'].data[...][0,0,:,:]
	    loss_list.append(loss)

	    density_map = cv2.imread(ds.batch_density_path_list[0], 0)
	    # evaluate  metric  cc, sim
	    cc =  CC(prediction, density_map)
	    sim = SIM(prediction, density_map)
	    cc_list.append(cc)
	    sim_list.append(sim)



	    # record loss and metric

    loss = np.mean(loss_list) 
    cc =  np.mean(cc_list)  
    sim =  np.mean(sim_list)  
    #record loss and metric