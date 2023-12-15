# load sources

import matplotlib
import os,sys

import numpy as np
import matplotlib.pylab as plt

from astropy import units as u
from astropy import constants as c 
from astropy.io import fits
from astropy.table import Table
from scipy import signal, interpolate

import resample

SPEEDOFLIGHT = 299792.458 # km/s

def vac_to_stand(wave_vac):
    """Convert vacuum wavelength (Ang) to standard wavelength in air since we're
    doing ground based stuff. 

    https://idlastro.gsfc.nasa.gov/ftp/pro/astro/vactoair.pro
    Equation from Prieto 2011 Apogee technical note
    and equation and parametersfrom Cidor 1996
    
    inputs: 
    -------
    wave_fac: 1D array, wavelength [A]

    outputs:
    -------
    
    """
    # eqn
    sigma2= (1e4/wave_vac)**2.
    fact = 1. +  5.792105e-2/(238.0185 - sigma2) + \
                            1.67917e-3/( 57.362 - sigma2)
                            
    # return l0 / n which equals lamda
    return wave_vac/fact

def load_phoenix_spectrum(T,wavbounds=None,datapath=None,norm=False):
    """
    Load phoenix spectrum

    input:
    ------
    T: [K] int
        temp of star (choices: 2300,4100,5800)
    R: [l/dl] float
        resolving power of spectrograph
    nsamp: [pixels] float
        pixel sampling of spectrograph
    wavbounds: [nm] tuple of length 2 or None
        [lam0,lam1] defines subset wavelength bounds of mask to load, or None to load whole array
    datapath: str
        path to data folder with mask

    returns:
    -------
    lam - [nm] array
        wavelengths
    spec - [phot/m2/s/nm] array
        spectral flux
    """
    teff     = str(int(T)).zfill(5)
    stelname = 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff)

    f = fits.open(datapath + stelname)
    spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
    f.close()
    
    path = stelname.split('/')
    f = fits.open(datapath + \
                     'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    lam = f[0].data # angstroms
    f.close()
    
    # Convert to photon flux
    conversion_factor = 5.03*10**7 * lam #lam in angstrom here
    spec *= conversion_factor # phot/cm2/s/angstrom
    
    if wavbounds != None:
        wav_start, wav_end = wavbounds
        # Take subarray requested
        isub = np.where( (lam > (wav_start)*10.0) & (lam < (wav_end)*10.0))[0]
        lam, spec = lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm
    else:
        lam = lam/10

    # reinterpolate to finer sampling
    tck = interpolate.splrep(lam,spec, k=5, s=0)
    finer_lam = np.arange(np.min(lam),np.max(lam),0.0001)
    finer_spec = interpolate.splev(finer_lam,tck,der=0,ext=1)

    if norm: finer_spec /= np.max(finer_spec)
 
    return finer_lam, finer_spec 

def shift_spectrum(lam,spec,v,order=3):
    """
    shift spectrum by velocity (km/s)

    output:
    -------
    shifted_spec: arr
        spectrum shifted by velocity, v

    """
    doppler_shift = (1.0 + (v / SPEEDOFLIGHT))
    tck = interpolate.splrep(lam*doppler_shift,spec, k=order, s=0)

    shifted_spec = interpolate.splev(lam,tck,der=0,ext=1)

    return shifted_spec

def scale_spec():
	"""
	Scale Spectrum to be the right SNR and Resample

	inputs
	------
	snr: float, signal to noise ratio of a resolution element in KPF

	outputs:
	--------
	kpf_spec: 1D array, phoenix spectrum scaled to input snr
	"""
	# times exp time, telescope aperture, transmission
	s_ccd_hires = so.stel.s * so.var.exp_time * so.const.tel_area * so.kpf.ytransmit * np.abs(so.tel.s)**1.1547

	# convolve to lower res
	s_ccd_lores = degrade_spec(so.stel.v, s_ccd_hires, so.const.res_kpf)

	s_order_avg = np.zeros_like(so.kpf.order_wavelengths)
	s_order_max = np.zeros_like(so.kpf.order_wavelengths)
	for i,wl in enumerate(so.kpf.order_wavelengths):
		#find order subset
		fsr = 1.61e-5 * wl**2 # empirical stab at fsr, too lazy for constants
		order_sub          = np.where(np.abs(so.stel.v - wl) < fsr/2)[0]
		# resample for that order
		sig                = wl/so.const.res_kpf/3.5 # lambda/res = dlambda, 5 pixel sampling
		v_resamp, s_resamp = resample(so.stel.v[order_sub],s_ccd_lores[order_sub],sig=sig, dx=0, eta=1,mode='fast')

		#plt.plot(v_resamp,np.sqrt(s_resamp))
		#average the order and store
		s_order_avg[i]     = np.sqrt(np.median(s_resamp))
		s_order_max[i]     = np.sqrt(np.max(s_resamp))

	return so.kpf.order_wavelengths, s_order_avg, s_order_max

def mod_spec(lam,spec,cont=None,v=0,R=100000,nsamp=3,order=5,norm=False):
    """
    shift to velocity, then resample after degrading to right spectral resolution
    
    inputs:
    ------
    nsamp [pixels] float
        pixel sampling of final spectrum

    outputs:
    -----
    """
    if v!=0:
        shifted_spec = shift_spectrum(lam,spec,v,order=order)
    else:
        shifted_spec = spec.copy()

    lowres_spec = resample.degrade_spec(lam,shifted_spec,R)
    
    if np.any(cont)==None:
        cont = np.ones_like(lowres_spec)

    sig = np.mean(lam)/R/nsamp
    if sig >= 2*np.mean(np.diff(lam)):
        final_lam,final_spec = resample.resample(lam,cont * lowres_spec,sig,mode='fast') # note this doesnt conserve flux very well because rounding of tophat - should really weight tophat and not round sig
        if norm: final_spec/=np.max(final_spec)
        return final_lam,final_spec
    elif sig < 2*np.mean(np.diff(lam)):
        final_lam = np.arange(np.min(lam), np.max(lam),sig)
        tck = interpolate.splrep(lam,cont*lowres_spec,k=order, s=0)
        shifted_spec = interpolate.splev(lam,tck,der=0,ext=1)
        print('NOT RESAMPLING')
    else:
        # should have option here to increase sampling by interpolating 
        final_spec = cont*lowres_spec
        if norm: final_spec/=np.max(final_spec)
        return lam, final_spec



if __name__=='__main__':
	pass




