%matplotlib notebook
import numpy as np
from scipy.optimize import leastsq, brent
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from time import process_time
from numpy.linalg import inv
np.set_printoptions(precision=16)
from Imaging_core_new import *
from Gridding_core import *
import pickle
with open("min_misfit_gridding_7.pkl", "rb") as pp:
    opt_funcs = pickle.load(pp)

#########  Read in visibilities ##########
data = np.genfromtxt('simul3d.csv', delimiter = ',')
jj = complex(0,1)
u_original = data.T[2][1:]
v_original = data.T[3][1:]
w_original = data.T[4][1:]
V_original = data.T[5][1:] + jj*data.T[6][1:]
n_uv = len(u_original)
uv_max = max(np.sqrt(u_original**2+v_original**2))
V,u,v,w = Visibility_minusw(V_original,u_original,v_original,w_original)

#### Determine the pixel size ####
X_size = 300 # image size on x-axis
Y_size = 300 # image size on y-axis
X_min = -1.15/2 #You can change X_min and X_max in order to change the pixel size.
X_max = 1.15/2
X = np.linspace(X_min, X_max, num=X_size+1)[0:X_size]
Y_min = -1.15/2 #You can change Y_min and Y_max in order to change the pixel size.
Y_max = 1.15/2
Y = np.linspace(Y_min,Y_max,num=Y_size+1)[0:Y_size]
pixel_resol_x = 180. * 60. * 60. * (X_max - X_min) / np.pi / X_size
pixel_resol_y = 180. * 60. * 60. * (Y_max - Y_min) / np.pi / Y_size
print ("The pixel size on x-axis is ", pixel_resol_x, " arcsec") 

W = 7
x0=0.25
Nw_2R, w_values, dw = Wplanes(W, X_max, Y_max, w, x0)

Nfft = 600
im_size = 600
M, x0, h = opt_funcs[W].M, opt_funcs[W].x0, opt_funcs[W].h
ind = find_nearestw(w_values, w)
C_w = cal_grid_w(w, w_values, ind, dw, W, h, M)

V_wgrid, u_wgrid, v_wgrid, beam_wgrid = grid_w(V, u, v, w, C_w, w_values, W, Nw_2R, ind)

I_size = int(im_size*2*x0)
I_image = np.zeros((I_size,I_size),dtype = np.complex_)
B_image = np.zeros((I_size,I_size),dtype = np.complex_)

t2_start = process_time() 
for w_ind in range(Nw_2R):
    print ('Gridding the ', w_ind, 'th level facet out of ',Nw_2R,' w facets.\n')
    V_update = np.asarray(V_wgrid[w_ind])
    u_update = np.asarray(u_wgrid[w_ind])
    v_update = np.asarray(v_wgrid[w_ind])
    beam_update = np.asarray(beam_wgrid[w_ind])
    V_grid, B_grid = grid_uv(V_update, u_update, v_update, beam_update, W, im_size, X_max, X_min, Y_max, Y_min, h, M)
    print ('FFT the ', w_ind, 'th level facet out of ',Nw_2R,' w facets.\n')
    I_image += FFTnPShift(V_grid, w_values[w_ind], X, Y, im_size, x0)
    B_image += FFTnPShift(B_grid, w_values[w_ind], X, Y, im_size, x0)
    B_grid = np.zeros((im_size,im_size),dtype = np.complex_) 
    V_grid = np.zeros((im_size,im_size),dtype = np.complex_)
    
t2_stop = process_time()   
print("Elapsed time during imaging in seconds:", t2_stop-t2_start)  

I_image_now = image_rescale(I_image,im_size, n_uv)
B_image_now = image_rescale(B_image,im_size, n_uv)
plt.figure()
plt.imshow(np.rot90(I_image_now.real,1), origin = 'lower')
plt.xlabel('Image Coordinates X')
plt.ylabel('Image Coordinates Y')
plt.show()
B_image_now[150,150]

Nfft = 600
# Use these for calculating gridding correction on the FFT grid
M = 32
I_xycorrected = xy_correct(I_image_now, opt_funcs[W], im_size, x0=0.25)
B_xycorrected = xy_correct(B_image_now, opt_funcs[W], im_size, x0=0.25)

Cor_gridz = z_correct_cal(X_min, X_max, Y_min, Y_max, dw, h, im_size, W, M, x0)
I_zcorrected = z_correct(I_xycorrected, Cor_gridz, im_size, x0=0.25)
B_zcorrected = z_correct(B_xycorrected, Cor_gridz, im_size, x0=0.25)

I_DFT = np.loadtxt('I_DFT_simul300.csv', delimiter = ',')

I_dif = I_DFT - I_zcorrected.real
rms = RMS(I_dif, im_size, 0.5, x0=0.25)
plt.figure()
plt.imshow(np.rot90(I_dif,1), origin = 'lower')
plt.colorbar()
plt.xlabel('Image Coordinates X')
plt.ylabel('Image Coordinates Y')
plt.show()