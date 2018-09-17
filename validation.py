import os
import cv2
import numpy as np
from saliency_metric.benchmark.metrics import *




def validation(solver_instance, dataset_instance, snapshot_dir, stops):
    loss_list = []
    cc_list = []
    sim_list = []
    validation_directory = os.path.join(snapshot_dir, 'validation_output')
    if not os.path.isdir(validation_directory):
    	os.mkdir(validation_directory)
    # print dir(solver_instance), solver_instance.snapshot();exit()
    while dataset_instance.completed_epoch < 1:
        frame_minibatch, _ = dataset_instance.next_hdr_batch(1,stops=stops)

        solver_instance.net.blobs['data'].data[...] = frame_minibatch
        # solver_instance.net.blobs['gt'].data[...] = density_minibatch
        solver_instance.net.forward()
        loss = solver_instance.net.blobs['loss'].data[...].tolist()[0]
        prediction = solver_instance.net.blobs['predict'].data[...][0,0,:,:]
        loss_list.append(loss)


        prediction = prediction-np.min(prediction)
        prediction = prediction/np.max(prediction)
        prediction = prediction * 255
        # print prediction[0]
        density_map = cv2.imread(dataset_instance.batch_density_path_list[0], 0).astype(np.float32)
        img_name = os.path.basename(dataset_instance.batch_density_path_list[0]).split('.')[0]
        img_path = os.path.join(validation_directory, img_name+'.jpg')


        cv2.imwrite(img_path, prediction)


        # evaluate  metric  cc, sim
        # cc_list.append(CC(prediction, density_map))
        # sim_list.append(SIM(prediction, density_map))
    # dataset_instance.completed_epoch=0 # reset epoch counter
    # print loss_list
    loss = np.mean(loss_list) 
    print "validation loss:", loss
    # cc =  np.mean(cc_list)  
    # sim =  np.mean(sim_list)  
    #record loss and metric