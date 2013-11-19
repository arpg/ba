import numpy as np
from matplotlib import *
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

    # this index points to the current location of the marginalized items in
    # the hessian
    pointer_idx = num_elems;

    # get the index of the pose in the hessian
    pose_idx = pose_id * pose_size

    # create the from/to indices for moving the pose columns/rows to the
    # bottom right of the matrix
    from_ids = np.arange(pose_idx, pose_idx + pose_size)
    to_ids = np.arange(pointer_idx - pose_size, pointer_idx)

    # swap the rows/columns in the hessian
    hessian[:, [from_ids, to_ids]] = hessian[:, [to_ids, from_ids]]
    hessian[[from_ids, to_ids], :] = hessian[[to_ids, from_ids], :]

    # update the pointer after the pose movement
    pointer_idx -= pose_size

    # go through each landmark and move it to the appropriate location
    for lm_id in lm_ids:
        lm_idx = pose_elems + lm_id * lm_size
        # create the from/to arrays for the move
        from_ids = np.arange(lm_idx, lm_idx + lm_size)
        to_ids = np.arange(pointer_idx - lm_size, pointer_idx)

        # swap the rows/columns in the hessian
        hessian[:, [from_ids, to_ids]] = hessian[:, [to_ids, from_ids]]
        hessian[[from_ids, to_ids], :] = hessian[[to_ids, from_ids], :]

        # update the pointer
        pointer_idx -= lm_size

    # once we have the pose/landmars in the correct spot, we want to perform
    # the schur complement: U - W*V^-1*W
    schur_elems = num_elems - (pose_size + lm_ids.shape[0] * lm_size)
    # U = hessian[0 : num_elems, 0 : num_elems]
    W = hessian[0 : num_elems, num_elems : ]
    V = hessian[num_elems : , num_elems :]
    prior = W.dot(inv(V)).dot(W.T);





