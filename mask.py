import numpy as np
from scipy import signal
import matplotlib.pylab as plt
from scipy import interpolate
from astropy.io import fits
import glob

plt.ion()
plt.rcParams['font.size'] = '12'

def make_telluric_mask(lam,spec,cutoff=0.01,velocity_cutoff=30,water_only=False):
    """
    spec: flat spec, no continuum changes
          should already be at resolution care about
    """
    spec[np.where(np.isnan(spec))] = 0
    
    #cutoff = 0.01 # reject lines greater than 1% depth
    telluric_mask = np.ones_like(spec)
    telluric_mask[np.where(spec < (1-cutoff))[0]] = 0
    # avoid +/-5km/s  (5pix) around telluric
    for iroll in range(velocity_cutoff):
        telluric_mask[np.where(np.roll(spec,iroll) < (1-cutoff))[0]] = 0
        telluric_mask[np.where(np.roll(spec,-1*iroll) < (1-cutoff))[0]] = 0

    return telluric_mask#,spec



def make_UNe_mask(shorten=False):
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

    if shorten: # output a subset
        Nlines = len(mask_cen)//2 - 5
        return mask_cen[0:Nlines], mask_wid[0:Nlines], mask_weights[0:Nlines]
    
    if save:
        np.savetxt(filepath + 'une_ccfmask.fits',np.vstack(mask_cen,mask_weights),\
            header='mask_cen mask_weight')
    
    return mask_cen, mask_weights,path



def make_phoenix_mask(teff=3800,wav_start=900,wav_end=2500,height=0.01,prominence=0.01,distance=1,plot=False,save=True):
    """
    load phoenix spectrum and find minima to get mask positions and weights
    
    spectrum source: https://phoenix.astro.physik.uni-goettingen.de/?page_id=15
    inputs
    ------
    teff - [K] 
        temperature of Phoenix model (must be present in path) 
    wav_start - float [nm] 
        start wavelength of phoenix model to make mask for 
    wav_end - float [nm] 
        end  wavelength of phoenix model to make mask for 
    height - float [0,1] 
        height of a peak to consider (no continuum correction applied so keep it low)
    prominence - float [0,1] 
        prominence of peak to consider
    plot - bool
        if True will plot the spectrum and the peaks
    save - bool
        if True will save the mask center and heights to csv file in same folder as data

    outputs
    ------
    mask_cen     - [int] wavelength of peak of spectral line
    mask_weights - [float] peak of line at position x

    """
    # LOAD PHOENIX MODEL
    filepath = '/Users/ashbake/Documents/Research/Projects/HISPEC/RV_simulations/rvsim/data/1DSpectra/Phoenix/'
    teff     = str(int(teff)).zfill(5)
    stelname = 'lte%s-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'%(teff)

    f = fits.open(filepath + stelname)
    spec = f[0].data / (1e8) # ergs/s/cm2/cm to ergs/s/cm2/Angstrom for conversion
    f.close()
    
    path = stelname.split('/')
    f = fits.open(filepath + \
                     'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')
    lam = f[0].data # angstroms
    f.close()
    
    # Convert to photon flux
    conversion_factor = 5.03*10**7 * lam #lam in angstrom here
    spec *= conversion_factor # phot/cm2/s/angstrom
    
    # Take subarray requested
    isub = np.where( (lam > wav_start*10.0) & (lam < wav_end*10.0))[0]
    wave, spec = lam[isub]/10.0,spec[isub] * 10 * 100**2 #nm, phot/m2/s/nm

    # interpolate to get spectrum sampled better
    tck = interpolate.splrep(wave,spec, k=3, s=0)
    wave_high_sampling = np.arange(wav_start,wav_end,0.0001)
    spec_high_sampling = interpolate.splev(wave_high_sampling,tck,der=0,ext=1)

    # FIND PEAKS
    f = 1 - spec_high_sampling/np.max(spec_high_sampling) #flip it so troughs are peaks, normalize 0-->1
    peaks, heights = signal.find_peaks(f,height=height,prominence=prominence,distance=distance)

    mask_cen,mask_weights = wave_high_sampling[peaks], f[peaks]

    if plot:
        plt.figure('Peak Finder Results',figsize=(8,5))
        plt.plot(wave,1 - spec/np.max(spec),label='Phoenix, Teff=%s'%teff)
        #plt.plot(wave_high_sampling,f,label='Spline Reinterpolated to Higher Sampling')
        plt.plot(wave_high_sampling[peaks],f[peaks],'o',label='Identified Mask Lines')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('1 - Normalized Flux')
        plt.legend(fontsize=10)
        plt.grid()

    if save:
        np.savetxt(filepath + stelname.replace('.fits','_ccfmask.csv'),np.vstack((mask_cen,mask_weights)).T,\
            header='mask_cen mask_weight')

    return mask_cen, mask_weights,filepath


def load_phoenix_mask(T=2300,wavbounds=None,datapath='./rvsim/data/Masks/'):
    """
    load phoenix mask, assumes filename ends in 'ccfmask.csv' and includes temp

    input:
    ------
    T: [K] int
        temp of star (choices: 2300,4100,5800)
    wavbounds: [nm] tuple of length 2 or None
        [lam0,lam1] defines subset wavelength bounds of mask to load, or None to load whole array
    datapath: str
        path to data folder with mask

    returns:
    -------
    mask cen - [nm] array
        mask center positions
    mask weight - [0,1] array
        mask weights corresponding to each center position
    """
    files = glob.glob(datapath + '*%s*ccfmask.csv' %T)
    f = np.loadtxt(files[0])
    mask_cen, mask_weight = f[:,0],f[:,1]

    if np.any(wavbounds) == None:
        return mask_cen, mask_weight 
    else:
        bounds = np.where((mask_cen > wavbounds[0]) &\
                           (mask_cen < wavbounds[1]))[0]
        if len(bounds) == 1:
            raise Warning("Check wavelength bounds provided fall within mask wavelength range")
        return mask_cen[bounds], mask_weight[bounds]


if __name__=='__main__':
    # make masks, one function for each type of spectrum
    #mask_cen, mask_weight, filepath = load_UNe_spec() 
    mask_cen, mask_weight, filepath = make_phoenix_mask(teff=4100,
                                                        wav_start=350,
                                                        wav_end=2500,
                                                        prominence=0.01,
                                                        height=None,
                                                        distance=1,
                                                        plot=True,
                                                        save=True)
    


