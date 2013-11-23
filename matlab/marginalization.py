import numpy as np
def marginalize_pose(hessian, num_poses, pose_size,
                     num_lm, lm_size, pose_id, lm_ids):
    """
    Given a hessian, this function marginalizes a single pose and its given
    landmarks and returns a prior matrix which represents the marginalized
    constraints

    :param hessian: The complete hessian matrix
    :param num-poses: The number of poses in the hessian
    :param pose_size: The dimension of each pose (e.g. 6, 9)
    :param num_lm: The number of landmarks in the hessian
    :param lm_size: The dimension of each landmark (e.g. 1, 3)
    :param pose_id: The index of the pos to be marginalized
    :param lm_ids: The indices of the landmarks to be marginalized
    """
    # get the number of rows and columns of the hessian
    num_elems = hessian.shape[0]
    pose_elems = num_poses * pose_size
    
    # first shift the landmarks to the end
    lm_cols = hessian[:, lm_ids + pose_elems];
    hessian = np.delete(hessian, lm_ids + pose_elems, 1);
    hessian = np.lib.index_tricks.c_[hessian, lm_cols];
    
    lm_rows = hessian[lm_ids + pose_elems, :];
    hessian = np.delete(hessian, lm_ids + pose_elems, 0);
    hessian = np.lib.index_tricks.r_[hessian, lm_rows];

    # get the index of the pose in the hessian
    pose_idx = pose_id * pose_size
    
    #get the index of where the pose will end up
    final_pose_idx = num_elems - pose_size;

    # swap the rows/columns in the hessian
    idx = np.arange(pose_idx, num_elems)
    hessian[:, idx] = np.roll(hessian[:, idx], 
                             final_pose_idx - pose_idx, 1)
    hessian[idx, :] = np.roll(hessian[idx, :], 
                             final_pose_idx - pose_idx, 0)

    # once we have the pose/landmars in the correct spot, we want to perform
    # the schur complement: U - W*V^-1*W
    schur_elems = num_elems - (pose_size + lm_ids.shape[0] * lm_size)
    # U = hessian[0 : num_elems, 0 : num_elems]
    W = hessian[0 : schur_elems, schur_elems : ]
    V = hessian[schur_elems : , schur_elems :]
    prior = W.dot(np.linalg.inv(V)).dot(W.T);
    pass





