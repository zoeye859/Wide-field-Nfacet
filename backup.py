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
        C_w += [calc_C(h, x0, np.asarray([tempw]), W)]
    t_stop = process_time()   
    print("Elapsed time during the w gridding value calculation in seconds:", t_stop-t_start)  
    return C_w

def find_nearestw(w_values, w):
    """
    For each w value, find the index of the nearest w plane that this w value would be gridded to
    Args:
        w_values (list): w values for all w-planes would be formed
        w (float): w of the (u,v,w) coordinates
    Returns:
        idx (list): the index of the nearest w plane that this w value would be gridded to
    """
    w_values = np.asarray(w_values)
    idx= np.searchsorted(w_values, w, side="left")
    return idx-1
