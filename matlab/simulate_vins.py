'''
Created on Dec 4, 2013

@author: nimski
'''
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def enum(**enums):
    return type('Enum', (), enums)

PathTypes = enum(CURVY_SQUARE=1, CURVY_WALK=2)

def cart_2_r(x):
    cr = sp.cos(x[0])
    cp = sp.cos(x[1])
    cq = sp.cos(x[2])
    sir = sp.sin(x[0])
    sip = sp.sin(x[1])
    siq = sp.sin(x[2])
    return sp.Matrix([[cp*cq, -cr*siq+sir*sip*cq, sir*siq+cr*sip*cq ],
                     [cp*siq, cr*cq+sir*sip*siq, -sir*cq+cr*sip*siq ],
                     [-sip, sir*cp, cr*cp ]])

def make_t(R,t):
    t3_4 = R.row_join(t)
    return t3_4.col_join(sp.Matrix([[0, 0, 0, 1]]))

def cart_2_t(x):
    return make_t(cart_2_r([x[3],x[4],x[5]]), sp.Matrix( [x[0],x[1],x[2]] ) )

def t_inv(T):
    r_inv = T[0:3, 0:3].transpose();
    return make_t( r_inv, -r_inv * T[0:3, 3] )

def r_vee(R):
    return sp.Matrix( [R[2,1], R[0,2], R[1,0]] )

def diff_matrix(m, param):
    m_out = sp.Matrix.zeros(m.shape[0], m.shape[1])
    for x in range(0, m_out.shape[0]):
        for y in range(0, m_out.shape[1]):
            m_out[x,y] = sp.diff(m[x, y], param)
            
    return m_out
    

def generate_trajectory(type):
    # First generate the imu trajectory.
    # Curvy square.
    max_time = 3 * 2 * np.pi
    step = max_time / 2000
    g = sp.Matrix([[0], [0], [-9.806]])
    
    t = sp.Symbol('t')
    expr = 7.5 + sp.sin(4 * t - np.pi / 2) / 2 
        
    if type == PathTypes.CURVY_SQUARE:
        trajectory = ([expr.subs(t, np.pi + t / 3) * sp.cos(np.pi + t / 3), 
                    expr.subs(t, np.pi + t / 3) * sp.sin(np.pi + t / 3), 
                    -1.5,   
                    0, 
                    0, 
                    3 * np.pi / 2 + t / 3])
        
    elif type == PathTypes.CURVY_WALK:
        trajectory = sp.Matrix([expr.subs(t, np.pi + t / 3) * sp.cos(np.pi + t / 3), 
                       expr.subs(t, np.pi + t / 3) * sp.sin(np.pi + t / 3), 
                       -1.5 + 0.05 * sp.sin(10 * t) , 
                       np.pi / 2, 
                       0, 
                       np.pi / 2 + 
                       (sp.atan2( sp.diff(expr.subs(t, np.pi + t / 3) * 
                                          sp.sin(np.pi + t / 3),t) , 
                                  sp.diff(expr.subs(t, np.pi + t / 3) * 
                                          sp.cos(np.pi + t / 3),t))) ])
    
    
    # Calculate trajectory time derivatives.
    trajectory_mat = cart_2_t(trajectory)
    dtrajectory_mat = diff_matrix(trajectory_mat, t)
    ddtrajectory_mat = diff_matrix(dtrajectory_mat, t)
    
    # Transform the derivatives to body frame.
    dtrajectory_b = t_inv(trajectory_mat) * dtrajectory_mat
    permutation = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]]);
    trajectory_gyro = (permutation * r_vee(dtrajectory_b[0:3, 0:3]))
    trajectory_accel = ( permutation *  
                         (trajectory_mat[0:3, 0:3].transpose() *
                         (ddtrajectory_mat[0:3, 3] + g)) )
    

    trajectory_lambda = sp.lambdify(t, trajectory, "numpy")
    t_vals = np.arange(0, max_time, step)
    trajectory_vals = trajectory_lambda(t_vals)
    
    gyro_lambda = sp.lambdify(t, trajectory_gyro, "numpy")
    gyro_vals = gyro_lambda(t_vals)
    
    accel_lambda = sp.lambdify(t, trajectory_accel[0], "numpy")
    accel_vals = accel_lambda(t_vals)
        
    # Plot this curve.
    fig = plt.figure()
    ax = fig.gca(projection='3d', aspect='equal')
    ax.plot(trajectory_vals[0], trajectory_vals[1], 
            trajectory_vals[2], label='trajectory')
    plt.show()
    pass

generate_trajectory(PathTypes.CURVY_WALK) 