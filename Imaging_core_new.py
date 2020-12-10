import numpy as np
from time import process_time
from Gridding_core import *
np.set_printoptions(precision=16)


def Visibility_minusw(V,u,v,w):
    '''
    Conjugate Visibility data for each negative w, therefore the computational cost can be divided by 2
    Args:
        V (np.narray): visibility values
        u (np.narray): u of the (u,v,w) coordinates
        v (np.narray): v of the (u,v,w) coordinates
        w (np.narray): w of the (u,v,w) coordinates
    Returns:
        Conjugated visibilities
    '''
    for i in range(len(u)):
        if w[i]<0:
            #print (i)
            u[i] = -u[i]
            v[i] = -v[i]
            w[i] = -w[i]
            V[i] = np.conjugate(V[i])
    return V,u,v,w


def find_nearestw(w_values, w):
    """
    For each w value, find the index of the nearest w plane on either side that this w value would be gridded to
    Args:
        w_values (list): w values for all w-planes would be formed
        w (float): w of the (u,v,w) coordinates
    Returns:
        idx (list): the index of the nearest w plane that this w value would be gridded to
    """
    idx = []
    for i in range(len(w)):
        idx += [np.abs(w_values - w[i]).argmin()]
    return idx

def find_floorw(w_values, w):
    """
    For each w value, find the index of the nearest w plane on the left that this w value would be gridded to
    Args:
        w_values (list): w values for all w-planes would be formed
        w (float): w of the (u,v,w) coordinates
    Returns:
        idx (list): the index of the nearest w plane that this w value would be gridded to
    """
    w_values = np.asarray(w_values)
    idx= np.searchsorted(w_values, w, side="left")
    return idx-1

def cal_grid_uv(u, W, im_size, X_max, X_min, h, M, x0=0.25):
    """
    For each the given u values, find its W gridding weights
    
    Args:
        W (int): support width of the gridding function
        u (np.narray): u of the (u,v,w) coordinates
        im_size (int): the image size, it is to be noted that this is before the image cropping
        X_max (np.narray): largest X or l in radius, for v, it should be Y or m
        X_min (np.narray): minimum X or l in radius, for v, it should be Y or m
    Returns:
        C_u (list): the list of gridding weights for the u array
    """
    t_start = process_time() 
    u_grid = u * 2 * (X_max - X_min) + im_size//2
    C_u = []
    for k in range(len(u)):
        tempu = u_grid[k] - np.floor(u_grid[k])
        C_u += [calc_C(h, x0, np.asarray([tempu]), W)]
    t_stop = process_time()   
    print("Elapsed time during the u/v gridding value calculation in seconds:", t_stop-t_start)  
    return C_u, u_grid

def cal_grid_w(w, w_values, idx, dw, W, h, M, x0=0.25):
    """
    For each the given w values, find its W gridding weights
    
    Args:
        W (int): support width of the gridding function
        w (np.narray): w of the (u,v,w) coordinates
        w_values (list): w values for all w-planes would be formed
        idx (list): the index of the nearest w plane that this w value would be assigned to
        dw (float): difference between two neighbouring w-planes
        h (np.ndarray): The vector of grid correction values sampled on [0,x0) to optimize
    Returns - usually given
        C_w (list): the list of gridding weights for the w array
    """
    t_start = process_time() 
    C_w = []
    for k in range(len(w)):
        tempw = (w[k] - w_values[idx[k]])/dw
        C_w += [calc_C(h, x0, np.asarray([tempw+0.5]), W)]
    t_stop = process_time()   
    print("Elapsed time during the w gridding value calculation in seconds:", t_stop-t_start)  
    return C_w


def grid_w(V, u, v, w, C_w, w_values, W, Nw_2R, idx):
    """
    Grid on w-axis
    Args:
        V (np.narray): visibility data
        u (np.narray): u of the (u,v,w) coordinates
        v (np.narray): v of the (u,v,w) coordinates
        w (np.narray): w of the (u,v,w) coordinates
        Nw_2R (int): number of w-planes used
        W (int): support width of the gridding function
        w_values (list): w values for all w-planes would be formed
        idx (list): the index of the nearest w plane that this w value would be assigned to
        dw (float): difference between two neighbouring w-planes
        C_w (list): the list of gridding weights for the w array
    """
    n_uv = len(V)
    bEAM = np.ones(n_uv)
    V_wgrid = np.zeros((Nw_2R,1),dtype = np.complex_).tolist()
    beam_wgrid = np.zeros((Nw_2R,1),dtype = np.complex_).tolist()
    u_wgrid = np.zeros((Nw_2R,1)).tolist()
    v_wgrid = np.zeros((Nw_2R,1)).tolist()
    t_start = process_time() 
    idx_floor = find_floorw(w_values, w)

    for k in range(n_uv):
        C_wk = C_w[k]
        if W % 2 == 1:
            w_plane = idx[k]
        else:
            w_plane = idx_floor[k]
        j = 0
        for n in range(-W//2+1,-W//2+1+W):
            V_wgrid[w_plane+n] += [C_wk[j,0] * V[k]]
            u_wgrid[w_plane+n] += [u[k]]
            v_wgrid[w_plane+n] += [v[k]]
            beam_wgrid[w_plane+n] += [C_wk[j,0] * bEAM[k]]
            j+=1

    for i in range(Nw_2R):
        del(V_wgrid[i][0])
        del(u_wgrid[i][0])
        del(v_wgrid[i][0])
        del(beam_wgrid[i][0])

    t_stop = process_time()   
    print("Elapsed time during the w-gridding calculation in seconds:", t_stop-t_start)   
    return V_wgrid, u_wgrid, v_wgrid, beam_wgrid

def image_crop(I, im_size, x0=0.25):
    """
    Throw away the unwanted image part according to x_0.
    When x_0 = 0.25, we will crop the outer half of the image
    Args:
        I (np.narray): the original image
        im_size (int): the image size, it is to be noted that this is before the image cropping
        x_0 (float): central 2*x_0*100% of the image will be retained    w (np.narray): w of the (u,v,w) coordinates
    Returns:
        I_cropped (np.narray): the cropped image
    """
    I_size = int(im_size*2*x0)
    index_x = int(I_size * 1.5)
    index_y = int(I_size * 1.5)
    temp = np.delete(I,np.s_[0:I_size//2],0)
    temp = np.delete(temp,np.s_[I_size:index_x],0)
    temp = np.delete(temp,np.s_[0:I_size//2],1)
    return np.delete(temp,np.s_[I_size:index_y],1)


def FFTnPShift(V_grid, ww, X, Y, im_size, x0=0.25):
    """
    FFT the gridded V_grid, and apply a phaseshift to it
    Args:
        V_grid (np.narray): gridded visibility on a certain w-plane
        ww (np.narray): the value of the w-plane we are working on at the moment
        im_size (int): the image size, it is to be noted that this is before the image cropping
        x_0 (float): central 2*x_0*100% of the image will be retained    
        X (np.narray): X or l in radius on the image plane
        Y (np.narray): Y or m in radius on the image plane
    Returns:
        I (np.narray): the FFT and phaseshifted image
    """
    print ('FFTing...')
    jj = complex(0,1)
    I = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(V_grid)))
    I_cropped = image_crop(I, im_size)
    I_size = int(im_size*2*x0)
    I_FFTnPShift = np.zeros((I_size,I_size),dtype = np.complex_)
    print ('Phaseshifting...')
    for l_i in range(0,I_size):
        for m_i in range(0,I_size):
            ll = X[l_i]
            mm = Y[m_i]
            nn = np.sqrt(1 - ll**2 - mm**2)-1
            I_FFTnPShift[l_i,m_i] = np.exp(jj*2*np.pi*ww*nn)*I_cropped[l_i,m_i]
    return I_FFTnPShift


def image_rescale(I, im_size, n_uv):
    """
    Rescale the obtained image
    Args:
        I (np.narray): summed up image
        im_size (int): the image size, it is to be noted that this is before the image cropping
        n_uv (int): the number of visibibility data
    """
    return I*im_size*im_size/n_uv

def Wplanes(W, X_max, Y_max, w, x0=0.25):
    """
      Rescale the obtained image
    Args:
        I (np.narray): summed up image
        im_size (int): the image size, it is to be noted that this is before the image cropping
        n_uv (int): the number of visibibility data
    Return:
        Nw_2R (int): calculated the number of w-planes using the proposed new formula
        w_values (list): w values for all w-planes would be formed
        dw (float): difference between two neighbouring w-planes
    """
    N_w = int(np.ceil((1-np.sqrt(1-(X_max)**2-(Y_max)**2))*(np.max(w)-np.min(w))/x0))
    dw = (w.max() - w.min())/N_w
    left_idx = -W//2+1
    right_idx = np.abs(-W//2)+N_w
    w_values = [w.min() + dw * i for i in range(left_idx, right_idx)] # w vaule for each w-plane
    Nw_2R = len(w_values)
    print ("We will have", Nw_2R, "w-planes")   
    return Nw_2R, w_values, dw

def xy_correct(I, opt_func, im_size, x0=0.25):
    """
      Rescale the obtained image
    Args:
        W (int): support width of the gridding function
        im_size (int): the image size, it is to be noted that this is before the image cropping
        opt_func (np.ndarray): The vector of grid correction values sampled on [0,x0) to optimize
        I (np.narray): summed up image
    Return:
        I_xycorrected (np.narray): corrected image on x,y axis
    """ 
    I_size = int(im_size*2*x0)
    x = np.arange(-im_size/2, im_size/2)/im_size
    h_map = get_grid_correction(opt_func, x)
    index_x = int(I_size * 1.5)
    index_y = int(I_size * 1.5)
    temp = np.delete(h_map,np.s_[0:(im_size - index_x)],0)
    Cor_gridx = np.delete(temp,np.s_[I_size:index_x],0) #correcting function on x-axis
    Cor_gridy = np.delete(temp,np.s_[I_size:index_y],0) #correcting function on y-axis
    I_xycorrected = np.zeros([I_size,I_size],dtype = np.complex_)
    for i in range(0,I_size):
        for j in range(0,I_size):
            I_xycorrected[i,j] = I[i,j] * Cor_gridx[i] * Cor_gridy[j]
    return I_xycorrected

def int5(h_x, iz, zin, step):
    y0=h_x[iz]
    y1=h_x[iz+step]
    ym1=h_x[iz-step]
    y2=h_x[iz+2*step]
    y3=h_x[iz+3*step]
    ym2=h_x[iz-2*step]
    a0=y0
    if((zin<0.) or (zin>1.)):
        print("This should not happen\n.")
    else:
        a1 = y1-(1./3)*y0-(1./2)*ym1+(1./20)*ym2-(1./4)*y2+(1./30)*y3
        a2 = (2./3)*ym1-(5./4)*y0+(2./3)*y1-(1./24)*ym2-(1./24)*y2
        a3 = (7./24)*y2+(5./12)*y0-(7./12)*y1-(1./24)*ym2-(1./24)*ym1-(1./24)*y3
        a4 = (1./24)*ym2+(1./4)*y0-(1./6)*y1-(1./6)*ym1+(1./24)*y2
        a5 = (1./120)*y3-(1./12)*y0+(1./12)*y1+(1./24)*ym1-(1./24)*y2-(1./120)*ym2 
        ans = a0 + a1*zin + a2*zin*zin + a3*zin*zin*zin + a4*zin*zin*zin*zin +a5*zin*zin*zin*zin*zin
    return ans

def z_correct_cal(X_min, X_max, Y_min, Y_max, dw, h, im_size, W, M, x0):
    """
    Return:
        Cor_gridz (np.narray): correcting function on z-axis
    """ 
    I_size = int(im_size*2*x0)
    nu, x = make_evaluation_grids(W, M, I_size)
    gridder = calc_gridder(h, x0, nu, W, M)
    grid_correction = gridder_to_grid_correction(gridder, nu, x, W)
    h_map = np.zeros(im_size, dtype=float)
    h_map[I_size:] = grid_correction[:I_size]
    h_map[:I_size] = grid_correction[:0:-1]
    xrange = X_max - X_min
    yrange = Y_max - Y_min
    ny = im_size
    nx = im_size
    fmap = np.zeros((nx,ny))
    for i in range(ny):
        yy = 2.*yrange*(i - ny/2)/ny
        for j in range(nx):
            xx = 2.*xrange*(j - nx/2)/nx
            if (xx*xx + yy*yy > 0.99999999) or (abs(xx) > 0.55*xrange) or (abs(yy) > 0.55*yrange):
                z = 0.
            else:
                z = dw*(1. - np.sqrt(1. - xx*xx - yy*yy))
                ind0 = (int)(z*nx + nx/2.)
                xin = (float) (z*nx + nx/2.) - ind0
                fmap[i,j] = int5(h_map,ind0,xin,1)
    Cor_gridz = image_crop(fmap, im_size, x0)
    return Cor_gridz

def z_correct(I, Cor_gridz, im_size, x0=0.25):
    """
      Rescale the obtained image
    Args:
        W (int): support width of the gridding function
        im_size (int): the image size, it is to be noted that this is before the image cropping
        h (np.ndarray): The vector of grid correction values sampled on [0,x0) to optimize
        I (np.narray): summed up image
    Return:
        I_zcorrected (np.narray): corrected image on z-axis
    """ 
    I_size = int(im_size*2*x0)
    I_zcorrected = np.zeros([I_size,I_size],dtype = np.complex_)
    for i in range(0,I_size):
        for j in range(0,I_size):
            I_zcorrected[i,j] = I[i,j] * Cor_gridz[i,j]
    return I_zcorrected

def RMS(I_dif, im_size, area_percentage, x0=0.25):
    """
      Rescale the obtained image
    Args:
        I_dif (np.ndarray): DFT and FFT image difference
        im_size (int): the image size, it is to be noted that this is before the image cropping
        area_percentage (float): 1 for the whole map, 0.5 for the central half
    Return:
        rms (float): rms of the selected area of the difference map
    """ 
    I_size = int(im_size*2*x0)
    if area_percentage == 1:
        return np.sqrt((I_dif ** 2).mean())
    elif area_percentage > 1:
        print ('parameter area_percentage has to be equal or smaller than 1')
    else:
        idx = int(I_size * area_percentage/2)
        return np.sqrt((I_dif[idx:(I_size-idx)] ** 2).mean())