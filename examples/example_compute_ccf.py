import numpy as np
from astropy.modeling import models, fitting
from scipy import signal
import matplotlib.pylab as plt
plt.ion()
plt.rcParams['font.size'] = '12'

import ccf
from astropy.io import fits

# spectrum 1

R=100000
SPEEDOFLIGHT = 299792.458 # km/s
nsamp=3

def load_UNe_spec(shorten=True):
    """
    load UNe spectrum and find peaks to get mask positions and weights
    
    spectrum source: https://iopscience.iop.org/article/10.1088/0067-0049/199/1/2#apjs414945f5
    
    outputs
    ------
    mask_cen     - [int] wavelength of peak of spectral line
    mask_weights - [float] peak of line at position x

    """
    path = '/Users/ashbake/Documents/Research/Projects/HISPEC/RV_simulations/rvsim/data/1DSpectra/UNe_Redman2012/'
    wave = fits.getdata(path + 'une_wave.fits')/10 #nm
    spec = fits.getdata(path + 'une_spec.fits') #normalized flux

    peaks, heights = signal.find_peaks(spec/np.max(spec),height=0.005,prominence=0.003)

    mask_cen,mask_weights = wave[peaks], heights['peak_heights']

    width = 1
    mask_wid = np.ones_like(mask_cen) * width
    if shorten:
        Nlines = len(mask_cen)//2 - 5
        return mask_cen[0:Nlines], mask_wid[0:Nlines], mask_weights[0:Nlines]
    
    return mask_cen, mask_wid, mask_weights



def continuum_error():
    """
    code script dump, needs input
    """
    # add continuum to spectrum
    dy = -0.05 # 10percent
    i = 0 if dy > 0 else -1 # this prevents spec from being neg
    x0,y0= wlarr[i], fake_spec[i]
    continuum = dy * (wlarr - x0) + y0
    sloped_spec = fake_spec * continuum
    
    velocity_loop = np.arange(-0.5,0.5,5e-2) #km/s
    ccfout,_   = ccf.ccf_make(wlarr, sloped_spec, mask_cen, 1, mask_weight, velocity_loop)
    velfit     = ccf.fit_gaussian_to_ccf(velocity_loop, ccfout, 0, velocity_halfrange_to_fit=10,stddev_guess=0.3)

    # plot
    plt.figure('CCF and Fit Sloped Continuum')
    plt.title('(dy=%s) Velocity = %scm/s' %(dy,round(1e5 *velfit[0].mean_0.value,3) ))
    plt.plot(velocity_loop,ccfout,label='CCF')
    plt.plot(velocity_loop,velfit[0](velocity_loop),'--',c='lightgray',label='Best Fit')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('CCF')

    dys = np.array([-0.05, -0.02, -0.01, 0 , 0.01, 0.02, 0.05])
    vel = np.array([976.832, 1084.655, 1198.555, 1582.063, 2085.66, 2302.845, 2554.17 ])
    plt.plot(dys,vel-vel[3],'-o')
    plt.ylabel('Velocity Offset (cm/s)')
    plt.xlabel('Continuum Slope')

    # plot example sloped spec
    plt.figure('spectra display')
    for dy in dys:
        i = 0 if dy > 0 else -1
        x0,y0= wlarr[i], fake_spec[i]
        continuum = dy * (wlarr - x0) + y0
        plt.plot(wlarr, fake_spec * continuum,label='dy: %s'%dy)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.legend()

    # plot fake spectrum and mask
    #ccfout,mask0   = ccf.ccf_make(wlarr, fake_spec, mask_cen, 1, mask_weight, np.zeros(1))
    plt.figure('plot_mask')
    plt.plot(wlarr, fake_spec-0.5,label='Fake UNe Spectrum')
    plt.plot(wlarr, mask0,'k--',label='Mask')
    plt.xlabel('Wavelength (nm)')
    plt.legend()

if __name__=='__main__':
    # load real spectrum and get peak locations for mask and fake spectrum 
    mask_cen, mask_wid, mask_weight= load_UNe_spec() 

    # make fake spectrum at R of HISPEC
    wlarr = np.arange(np.min(mask_cen),np.max(mask_cen),(1/nsamp) *np.mean(mask_cen)/R)
    fake_spec = 1000 * ccf.spec_make(wlarr, mask_weight, mask_cen, mask_cen/R)

    # do the ccf
    velocity_loop = np.arange(-2,2,0.1) #km/s
    mask_wid = (SPEEDOFLIGHT / R) / nsamp #velocity_per_pix
    ccfout,mask   = ccf.ccf_make(wlarr, fake_spec, mask_cen, mask_wid, mask_weight, velocity_loop,R=R,nsamp=nsamp)
    velfit  = ccf.fit_gaussian_to_ccf(velocity_loop, ccfout, 0, velocity_halfrange_to_fit=10,stddev_guess=0.3)
    ccf_err = ccf.ccf_error_calc(velocity_loop, ccfout, fit_wid=mask_wid, pix_wid_vel=mask_wid)

    # plot mask
    plt.figure('plot_mask')
    plt.plot(wlarr, fake_spec-0.5,label='Fake UNe Spectrum')
    plt.plot(wlarr, mask,'-',label='Mask')
    plt.plot(wlarr, mask,'.',label='Mask')

    # plot
    plt.figure('CCF and Fit Flat Continuum')
    plt.title('Velocity = %scm/s' %round(1e5 *velfit[1].mean_0.value,3) )
    plt.plot(velocity_loop,ccfout-np.median(ccfout),label='CCF')
    plt.plot(velocity_loop,velfit[1](velocity_loop)-np.median(ccfout),'--',c='gray',label='Best Fit')
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('CCF')
    
    # get error - this is photon limited so depends on flux of spectrum (here it's 1 photonso this doesnt make sense)
    ccf.ccf_error_calc(velocity_loop, ccfout, fit_wid=100, pix_wid_vel=mask_wid)


