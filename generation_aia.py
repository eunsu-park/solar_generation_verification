import numpy as np

def solar_radius_cut(isize, rsun):
    X = np.arange(isize)[:, None]
    Y = np.arange(isize)[None, :]
    XY = np.sqrt((X-isize/2.)**2. + (Y-isize/2.)**2.)
    Z = np.where(XY < rsun)
    return Z

def pc(y_true, y_pred):
    """
    pixel-to-pixel correlation coefficient
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    result = np.corrcoef(y_true, y_pred)[0, 1]
    return result

def re(y_true, y_pred):
    """
    relative error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    flux_true = np.sum(y_true)
    flux_pred = np.sum(y_pred)
    result = (flux_pred - flux_true)/flux_true
    return result

def psnr(y_true, y_pred, max_i = 255):
    """
    peak signal to noise ratio
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.square(y_true-y_pred).mean()
    result = 20 * np.log10(max_i/np.sqrt(mse))
    return result

def nmse(y_true, y_pred):
    """
    normalized mean squared error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    result = np.square(y_pred-y_true).sum() / np.square(y_true).sum()
    return result

def ppe10(y_true, y_pred):
    """
    percentage of pixels having errors less than 10%
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ppe = np.abs(y_pred-y_true)/y_true
    result = ((np.where(ppe <= 0.1)[0].shape)[0]) / y_true.shape[0]
    return result

def rmscm(y_):
    """
    root mean squared contrast measure
    """
    y_ = np.array(y_)
    y_mean = np.mean(y_)
    result = np.square(y_-y_mean).mean()
    return result

    








    