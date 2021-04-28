def boundary_length(S):
    lx = S[1:,:]!=S[:-1,:]
    ly = S[:,1:]!=S[:,:-1]
    L = np.sum(lx)+np.sum(ly)
    return L

def boundary_loss(y_hat, y):
    """
    Assumes matrix input of dim [h,w]
    convertsion TENSOR.numpy()[0,:,:]
    
    Flipping:
    np.logical_not(y_hat.numpy()[0,:,:]).astype(int)
    """
    return abs(boundary_length(y_hat)-boundary_length(y))
