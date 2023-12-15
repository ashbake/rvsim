
import numpy as np
from astropy.modeling import models, fitting
import scipy.optimize as opt
from scipy import interpolate

from source import load_phoenix_spectrum, shift_spectrum, mod_spec
import resample

import matplotlib.pylab as plt
SPEEDOFLIGHT = 299792.458 # km/s

def model(p0, lam, spec, xmod, ymod, R,mode='chi2'):
    """
    """
    v0,A = p0
    shifted_spec = shift_spectrum(xmod,ymod,v0,order=5)

    lowres_spec = resample.degrade_spec(xmod,shifted_spec,R)

    # dont do flux conserving resample?? try it later to compare
    tck = interpolate.splrep(xmod,lowres_spec,k=5, s=0)
    final_spec = interpolate.splev(lam,tck,der=0,ext=1)

    if mode=='chi2':
        return np.sum((A*final_spec - spec)**2)
    else:
        return final_spec * A

def model_continuum(p0, lam, spec, xmod, ymod, R,knots, mode='chi2'):
    """
    include a continuum modification and a shift in this model
    """
    v0,cont_coeffs = p0[0],p0[1:]
    shifted_spec = shift_spectrum(xmod,ymod,v0,order=5)

    lowres_spec = resample.degrade_spec(xmod,shifted_spec,R)

    tck_cont = (knots, cont_coeffs, 3)
    continuum   = interpolate.splev(xmod, tck_cont)
    cont_spec   = continuum * lowres_spec

    # dont do flux conserving resample?? try it later to compare
    tck = interpolate.splrep(xmod,cont_spec,k=5, s=0)
    final_spec = interpolate.splev(lam,tck,der=0,ext=1)

    if mode=='chi2':
        return np.sum((final_spec - spec)**2)
    else:
        return final_spec


def setup_spline_continuum(x,y):
    """
    make spline model
    """
    n_interior_knots = 5
    qs = np.linspace(0, 1, n_interior_knots+2)[1:-1]
    knots = np.quantile(x, qs)

    # start by using best fit tck on line over range
    tck = interpolate.splrep(x,y,t=knots,k=3)
        
    # save knot points and coeff separate
    return tck[0],tck[1],tck[2]


def fit_spectrum_continuum(v0, lam, spec, xmod, ymod, R,method='Nelder-Mead'):
    '''
    fit spectrum for best RV where we take continuum
    Parameters
    ----------

    Returns
    -------
    '''
    # initiate guesses
    #spec_noisy = spec*(1 - (np.random.random(1)*.1 - 0.05))+ np.random.random(len(spec)) - 0.5
    knots, cont_coeffs, _ = setup_spline_continuum(lam,spec)
    cont_start = interpolate.splev(lam,(knots,cont_coeffs,3),der=0,ext=1)

    p0 = np.concatenate(([v0],cont_coeffs))

    # fit for v0 and amplitude
    out    = opt.minimize(model_continuum,p0,args=(lam,spec,xmod,ymod/np.max(ymod),R,knots),#,bounds=bounds,\
                                method=method,tol=1e-10)#,options={'maxiter' : 15000},tol=1e-20)

    bestfit = model_continuum(out['x'],lam,spec,xmod,ymod/np.max(ymod),R,knots,mode='spec')
    rv_fit = out['x'][0]
    chi2 = out['fun']

    return rv_fit, lam, spec, bestfit, chi2



def fit_spectrum(v0, lam, spec, xmod, ymod, R,method='Nelder-Mead'):
    '''
    Parameters
    ----------

    Returns
    -------
    '''
    # initiate guesses
    A  = 1.0
    p0 = [v0,A]

    # fit for v0 and amplitude
    out    = opt.minimize(model,p0,args=(lam,spec/np.max(spec),xmod,ymod/np.max(ymod),R),#,bounds=bounds,\
                                method=method,tol=1e-6)#,options={'maxiter' : 15000},tol=1e-20)

    bestfit = model(out['x'],lam,spec/np.max(spec),xmod,ymod/np.max(ymod),R,mode='spec')
    rv_fit = out['x'][0]
    chi2 = out['fun']

    return rv_fit, lam, spec/np.max(spec), bestfit, chi2



def ccf_fullspec(lam, spec, xmod, ymod, cont_mod,velocity_ccfloop):
    """
    cross correlate full spectrum
    """
    # pad spectra
    ccf = np.zeros_like(velocity_ccfloop)
    ccf2 = np.zeros_like(velocity_ccfloop)
    for i,v in enumerate(velocity_ccfloop):
        doppler_shift = (1.0 + (v / SPEEDOFLIGHT))
        tck = interpolate.splrep(xmod*doppler_shift,ymod,k=5, s=0)
        model_dopplershifted = cont_mod * interpolate.splev(lam,tck,der=0,ext=1)
        
        ccf[i] = np.nansum(spec * model_dopplershifted)
        ccf2[i] = np.trapz(y=spec * model_dopplershifted)

    return ccf, ccf2


