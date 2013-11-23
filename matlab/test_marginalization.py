# -*- coding: utf-8 -*-
import marginalization as mg
import numpy as np
def marginalize_pose_test():
    """
    Test function for testing the marginalize_pose function.
    """
    # create a temporary hessian
    hessian = np.arange(400).reshape(20, 20)
    num_poses = 4
    pose_size = 3
    num_lm = 8
    lm_size = 1
    pose_id = 1
    lm_ids = np.array([0, 1, 2])
    mg.marginalize_pose(hessian, num_poses, pose_size, num_lm, lm_size,
                        pose_id, lm_ids)

def marginalize_pose_csv_test():
    proj_pose_size = 6;
    pose_size = 9;
    u = np.genfromtxt("/Users/nimski/Code/Build/rslam/Applications/"
        "RelativeStereoSlam/u_orig.txt")
    w = np.genfromtxt("/Users/nimski/Code/Build/rslam/Applications/"
        "RelativeStereoSlam/w_orig.txt")
    # since w is always operating on 6 pose elements, we must pad it
    # horizontally
    w_padded = np.zeros([u.shape[0], w.shape[1]])
    c = np.array([range(i, i + proj_pose_size)
                 for i in range(0, u.shape[0])
                 if i % pose_size == 0]).flatten()
    w_padded[c , :] = w;

    v = np.genfromtxt("/Users/nimski/Code/Build/rslam/Applications/"
        "RelativeStereoSlam/v_orig.txt")
    # construct the hessian
    hessian = np.r_[np.c_[u, w_padded], np.c_[w_padded.T, v]]
    # the total number of poses in the hessian
    num_poses = u.shape[0] / pose_size
    # the pose id of the pose to be marginalized
    pose_id = num_poses - 1
    # we must now obtain the id of any landmark that was seen in this pos
    lm_ids = w_padded[pose_id * pose_size, :].nonzero()[0]
    mg.marginalize_pose(hessian, num_poses, pose_size, w.shape[1], 1 ,
                        num_poses-1, lm_ids)
    pass

# run the test
marginalize_pose_csv_test();