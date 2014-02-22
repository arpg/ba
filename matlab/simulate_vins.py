'''
Created on Dec 4, 2013

@author: nimski
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdastr


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

def project_linear(x_p, fx, fy, ux, uy):
    x_proj = np.array([x_p[0] / x_p[2], x_p[1] / x_p[2]])
    return np.array([x_proj[0] * fx + ux, 
                    x_proj[1] * fy + uy])
        
def project_fov(x_p, fx, fy, ux, uy, w):
    x_proj = np.array([x_p[0] / x_p[2], x_p[1] / x_p[2]])
    r = np.linalg.norm(x_proj)
    if (r < 1e-5):
        fac = 1
    else:
        mul2tanwby2 = 2.0 * np.tan(w / 2.0);
        mul2tanwby2byw = mul2tanwby2 / w;
        
        if(r * r < 1e-5):
            fac = mul2tanwby2byw
        else:
            fac = np.arctan(r * mul2tanwby2) / (r * w)
            
    return np.array([x_proj[0] * fac * fx + ux, 
                    x_proj[1] * fac * fy + uy])
 
    
def random_points_on_sphere(num_points, radius):
    points = np.random.rand(3,num_points)
    points = (points - 0.5) * radius
    points = np.array([points[0], points[1], points[2], np.ones(num_points)])
    return points

def generate_trajectory(type):
    fx = 198.969
    fy = 198.1284
    ux = 329.9368
    uy = 240.1017
    w = 0.9640582
    image_height = 480
    image_width = 640
    num_points = 500
    num_poses = 200
    gyro_sigma = 0.00104719755
    accel_sigma = 0.0392266
    pixel_sigma = 1.5
    bias_sigma = np.sqrt(1e-12)
    
    #gyro_bias = np.array([0.0675766003822892, 
    #                      0.0161325225711009, 
    #                      0.0381344050809982]) + np.random.normal(0, bias_sigma, 3)
    #accel_bias = np.array([-1.77133250613857,
    #                       -0.423571591333655,
    #                       -0.189646578628859])  + np.random.normal(0, bias_sigma, 3)

    gyro_bias = np.array([0, 0, 0]) 
    accel_bias = np.array([0, 0, 0])
                                                      
    print 'Gyro bias: ' + str(gyro_bias)
    print 'Accel bias: ' + str(gyro_bias)
    
    print 'Generating IMU trajectory.'
    
    # First generate the imu trajectory.
    # Curvy square.
    max_time = 3 * 2 * np.pi
    step = max_time / num_poses
    g = sp.Matrix([[0], [0], [-9.806]])
    
    t = sp.Symbol('t')
    expr = 7.5 + sp.sin(4 * t - np.pi / 2) / 2 
        
    if type == PathTypes.CURVY_SQUARE:
        trajectory = sp.Matrix([expr.subs(t, np.pi + t / 3) * sp.cos(np.pi + t / 3), 
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
    t_vals_imu = np.arange(0, max_time, step / 10)
    trajectory_vals = trajectory_lambda(t_vals)
    
    # remove any single length dimensions, this is required as some dimensions
    # do not change with time and sympy (stupidly) does not repeat them in 
    # the matrix, and so we must do so manually
    for i in range(0, 6):
        if trajectory_vals[i,0].__class__ != np.ndarray:
            trajectory_vals[i,0] = np.ones(len(t_vals)) * trajectory_vals[i,0] 
                    
    # get rid of sympy stupidness and place all values in a matrix
    trajectory_vals = np.r_[trajectory_vals[0,0].T,
                            trajectory_vals[1,0].T,
                            trajectory_vals[2,0].T,
                            trajectory_vals[3,0].T,
                            trajectory_vals[4,0].T,
                            trajectory_vals[5,0].T]
    
    gyro_lambda = sp.lambdify(t, trajectory_gyro, modules=({'ImmutableMatrix':np.array}, 'numpy'))
    
    gyro_vals = gyro_lambda(t_vals_imu[0])
    for i in range(1, len(t_vals_imu)):
        gyro_vals = np.c_[gyro_vals, gyro_lambda(t_vals_imu[i])]
        
            
    gyro_vals = np.c_[t_vals_imu,
                      gyro_vals[0],# + np.random.normal(0, gyro_sigma, len(t_vals_imu)) + gyro_bias[0],
                      gyro_vals[1],# + np.random.normal(0, gyro_sigma, len(t_vals_imu)) + gyro_bias[1],
                      gyro_vals[2]]# + np.random.normal(0, gyro_sigma, len(t_vals_imu)) + gyro_bias[2]]
    
    accel_lambda = sp.lambdify(t, trajectory_accel, modules=({'ImmutableMatrix':np.array}, 'numpy'))
    
    accel_vals = accel_lambda(t_vals_imu[0])
    for i in range(1, len(t_vals_imu)):
        accel_vals = np.c_[accel_vals, accel_lambda(t_vals_imu[i])]
    

    accel_vals = np.c_[t_vals_imu,
                       accel_vals[0],# + np.random.normal(0, accel_sigma, len(t_vals_imu)) * 0 + accel_bias[0],
                       accel_vals[1],# + np.random.normal(0, accel_sigma, len(t_vals_imu)) + accel_bias[1],
                       accel_vals[2]]# + np.random.normal(0, accel_sigma, len(t_vals_imu)) + accel_bias[2]]
    
    # create random points
    points = random_points_on_sphere(num_points, 80)
    
    # the matrix used to store the tracks
    tracks = np.empty((num_points * num_poses, 6,))
    
    count = 0
    print ('Projecting ' + str(points.shape[1]) + 
            ' landmarks into ' + str(len(t_vals)) + ' poses.') 
        
    # iterate over every pose
    for i in range(0, len(t_vals)): 
        T = trajectory_vals[:, i]
        time = t_vals[i]
        # transform the poses to the camera coordinate system
        points_proj = np.dot(np.linalg.inv(cart_2_t(T)), points)
        
        in_count = 0
        # project them into the camera
        for k in range(0, points_proj.shape[1]):
            point = points_proj[:, k]
            if point[2] > 0:
                in_count = in_count + 1
                point_proj = project_fov(point, fx, fy, ux, uy, w)
                #point_proj = project_linear(point, fx, fy, ux, uy)
                #point_proj = point_proj + np.random.normal(0, pixel_sigma, 2)
                
                if (point_proj[0] > 0 and point_proj[0]< image_width and
                    point_proj[1] > 0 and point_proj[1]< image_height):
                    tracks[count] = [time, i, k, 0, point_proj[0], point_proj[1]]
                    count = count + 1
                    
        print 'in count for pose ' + str(i) + ': ' + str(in_count)
                    
    print 'Saving csv files.'
                    
    # cull the unused rows of the tracks matrix
    tracks = tracks[0:count, :]
    np.savetxt("points.csv", tracks, fmt="%.12f,%d, %d, %d, %.12f, %.12f")
    np.savetxt("accel.csv", accel_vals, fmt="%.12f, %.12f, %.12f, %.12f")
    np.savetxt("gyro.csv", gyro_vals, fmt="%.12f, %.12f, %.12f, %.12f")
    np.savetxt("poses.csv", trajectory_vals.T, fmt="%.12f, %.12f, %.12f, %.12f, %.12f, %.12f")
    np.savetxt("timestamps.csv", t_vals, fmt="%.12f")
    
                     
                    
    print 'Plotting trajectory and landmarks.'
                        
    # Plot this curve.
    fig = plt.figure()
    ax = fig.gca(projection='3d', aspect='equal')
    ax.plot(trajectory_vals[0,:], trajectory_vals[1,:], trajectory_vals[2,:],
            label='trajectory')
    ax.scatter(points[0], points[1], points[2])
    plt.show()
    
    print 'Done.'
    pass

generate_trajectory(PathTypes.CURVY_SQUARE) 