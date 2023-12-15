'''
Function library for CCF generation / fitting, as well as photon-noise
estimates for fake spectra.
@authors: Arpita Roy, Sam Halverson
'''

import numpy as np
from astropy.modeling import models, fitting
import scipy.optimize as opt
from scipy import interpolate


import matplotlib.pylab as plt
# Set useful constants
SPEEDOFLIGHT = 299792.458 # km/s
GAUSSCONST = (2. * (2. * np.log(2))**0.5)

# 1D gaussian generation
def gaussian_fwhm(xarr, center, fwhm,A=1,B=0):
    '''
    Simple gaussian function, defined by center and FWHM

    Parameters
    ----------
    xarr : array
        Input dependant variable array
    center : float
        Center of gaussian distribution
    fwhm : float
        FWHM of gaussian desired
    A : float
        Amplitude of gaussian
    B : float
        Vertical offset of gaussian

    Returns
    -------
    gauss : array
        Computed gaussian values for xarr
    '''
    # gaussian function parameterized with FWHM
    gauss = A*np.exp(-0.5 * (xarr - center) ** 2. / (fwhm / GAUSSCONST) ** 2.) + B
    return gauss


# make spectra wrapper
def spec_make(wvl, weights, line_wvls, fwhms):
    '''
    Generate fake (normalized) spectrum of gaussian 'absorption' lines.

    Inputs:
    -------
    wvl : array
        Input wavelength array
    weights : array
        Line depths of specified lines
    line_wvls : array
        Line centers of features to be added
    fwhms : array
        FWHMs of lines specified

    Outputs:
    -------
    spec_out: array
         Final output absorption spectrum
    '''

    # initialize array
    spec_out = np.zeros_like(wvl)

    # for each line wavelength, add a gaussian at the specified depth
    for weight, line_wvl, fwhm in zip(weights, line_wvls, fwhms):
        spec_out += (weight * gaussian_fwhm(wvl, line_wvl, fwhm))
    return 1. - spec_out

# calculate photon-limited RV uncertainty (spectrum)
def spec_rv_noise_calc(wvl, spec):
    '''
    Calculates photon-limited RV uncertainty of given spectrum in km/s

    Parameters
    ----------
    wvl : array
        Input wavelength array of spectrum [nm]
    spec : array
        Flux values of spectrum -- assumes only photon noise

    Returns
    -------
    sigma_rv : float
        Computed photon-limited RV uncertainty [km/s]
    '''

    # calculate pixel optimal weights, follows Murphy et al. 2007
    wvl_m_ord = wvl * 1e-9 # convert wavelength values to meters

    # calculate noise (photon only, assume root N)
    sigma_spec = (spec)**0.5

    # calculate slopes of spectrum
    slopes = np.gradient(spec, wvl_m_ord)

    # calculate weighted slopes, ignoring the edge pixels (breaks derivative)
    top = (wvl_m_ord[1:slopes.size - 1]**2.) * (slopes[1:slopes.size - 1]**2.)
    bottom = (sigma_spec[1:slopes.size - 1]**2.)
    w_ord = top / bottom

    # combined weighted slopes
    return SPEEDOFLIGHT / ((np.nansum(w_ord))**0.5) # km/s

# add noise
def spec_add_noise(wvl, spec, wav_snr, snr):
    '''
    Scales and adds noise to a provided spectrum at a desired SNR value

    Parameters
    ----------
    wvl : :obj:'arr' of :obj:`float`
        Input wavlength array for spectrum [Ang or nm]

    spec : :obj:'arr' of :obj:`float`
        Flux array [photons]
    
    wav_snr : :obj:'float'
        Wavelength at which to scale to desired SNR [Ang or nm]

    snr : :obj:`float`
        Desired SNR to scale to

    Returns
    -------
    spec_noise : :obj:'arr' of :obj:`float`
        Scaled flux spectrum with noise added

    spec_scaled : :obj:'arr' of :obj:`float`
        Scaled flux spectrum with no noise added

    S Halverson - JPL - 29-Sep-2019
    '''
    spec = np.array(spec)
    wvl = np.array(wvl)

    # find index of wavelength array nearest to wav_snr
    ii = (np.abs(wvl - wav_snr)).argmin()
    ind = np.unravel_index(ii, spec.shape)
    #print(ind, spec[ind])

    # scale spectrum to specified SNR value
    spec_scaled = spec.copy() / spec[ind] # first normalize
    spec_scaled *= (snr ** 2.)

    # generate noise for given iteration
    noise_arr = np.random.standard_normal(spec_scaled.shape) * snr
        
    # add noise to frame (all orders at once)
    spec_noise = spec_scaled + noise_arr
    # spec_noise = spec_noise.clip(min=0) # get rid of negative numbers

    return spec_noise, spec_scaled

# compute CCF for given spectrum, mask
def ccf_make(wave, spectrum, mask_cen, mask_wid, mask_weight, velocity_loop,R=100000,nsamp=3):
    '''
    Generate cross-correlation function using weighted binary mask

    Parameters
    ----------
    wave : array
        wavelength array of target spectrum [Ang]
    spectrum : array
        flux spectrum of target
    mask_cen : array 
        wavelengths of mask lines used to compute CCF
    mask_wid : float [km/s]
        width of provided mask lines
    mask_weight : array
        weights of provided mask lines
    velocity_loop : array
        velocity step array [km/s]

    Returns
    ----------
    ccf : array
        computed cross-correlation function

    '''
    # initialize arrays
    v_steps = len(velocity_loop)
    ccf = np.zeros(v_steps)

    # Sort out mask left/right points
    line_start = np.array(mask_cen) * (1.0 - (mask_wid * 0.5 / SPEEDOFLIGHT))
    line_end = np.array(mask_cen) * (1.0 + (mask_wid * 0.5 / SPEEDOFLIGHT))
    line_center = np.array(mask_cen)
    line_weight = np.array(mask_weight)
    nlines = len(mask_cen)

    # Find pixel beginning and end wavelengths
    xpixel_wavestart = (wave + np.roll(wave, 1)) / 2.
    xpixel_waveend = np.roll(xpixel_wavestart, -1)

    # Fix edge pixel wavelengths
    xpixel_wavestart[0] = wave[0] - (wave[1]-wave[0]) / 2.
    xpixel_waveend[-1] = wave[-1] + (wave[-1]-wave[-2]) / 2.

    # Determine how many pixels to search on either side of central pixel
    # based on desired mask width and instrument specifications
    velocity_per_pix = (SPEEDOFLIGHT / R) / nsamp
    npix_search = (mask_wid) / velocity_per_pix
    npix_search = np.ceil(npix_search) + 1 #add an extra pixel for safety

    # Padding at the edges so search doesn't look for non-existent pixels
    npixels = np.shape(wave)[0]
    pix_start_limit = (npix_search + 2)
    pix_end_limit = npixels - (npix_search + 2)

    # Shift mask in redshift space
    doppler_shift = (1.0 + (velocity_loop / SPEEDOFLIGHT))

    # loop through velocity steps, generate binned mask spectrum
    for counter in range(v_steps):
        #print(str(counter) + ' out of %s'%v_steps)
        # new mask line wavelengths
        line_start_dopplershifted = line_start * doppler_shift[counter]
        line_end_dopplershifted = line_end * doppler_shift[counter]
        line_center_dopplershifted = line_center * doppler_shift[counter]
        closestmatch = np.sum((xpixel_wavestart - line_center_dopplershifted[:, np.newaxis] <= 0.),axis=1)

        # master array for shifted / rebinned mask spectrum
        mask_dopplershifted = np.zeros(npixels)

        # for each line, figure out how much weight is in each nearby pixel
        for ind_line in range(nlines):
            closest_xpixel = closestmatch[ind_line] - 1

            # get relevant nearest pixel values for beginning/end of mask
            lstart = line_start_dopplershifted[ind_line]
            lend = line_end_dopplershifted[ind_line]
            lweight = line_weight[ind_line]

            # if the nearest pixel is within the specified bounds,
            # fill in surrounding pixels with appropriate weights.
            if pix_start_limit < closest_xpixel < pix_end_limit:
                for ind_pix in range(int(closest_xpixel - npix_search), int(closest_xpixel + npix_search)):
                    if xpixel_wavestart[ind_pix] <= lend and xpixel_waveend[ind_pix] >= lstart:
                        wavestart = max(xpixel_wavestart[ind_pix], lstart)
                        waveend   = min(xpixel_waveend[ind_pix], lend)
                        mask_dopplershifted[ind_pix] = (lweight * (waveend - wavestart)
                                                        / (xpixel_waveend[ind_pix] -
                                                           xpixel_wavestart[ind_pix]))
        #plt.figure(-99)
        #plt.plot(wave,mask_dopplershifted)
        # fill in CCF with the sum of multipled spectrum
        ccf[counter] = np.nansum(spectrum * mask_dopplershifted)

    #plt.plot(wave,spectrum-0.4,'k')
    return ccf,mask_dopplershifted

# basic fit for CCF (Gaussian)
def fit_gaussian_to_ccf(velocity_loop, ccf, rv_guess, velocity_halfrange_to_fit=100.0,stddev_guess=0.3):

    '''
    Fit a Gaussian to the cross-correlation function

    Note that median value of the CCF is subtracted before
    the fit, since astropy has trouble otherwise.

    Analyses of CCF absolute levels must be performed separately.

    Parameters
    ----------
    velocity_loop : array
        Input velocity step array for CCF [km/s]
    rv_guess : float
        Initial guess for RV
    velocity_halfrange_to_fit : float
        Full range of CCF to be fit (must be narrower than velocity_loop)
    stddev_guess : float
        guess for standard deviation of fit to CCF
    
    Returns
    -------
    gaussian_fit : object
        Final Gaussian fit (evaluated)
    gx, gy: arrays
        Narrowed CCF arrays for window that was actually used for final fit
    rv_mean: float
        Final RV measurement [km/s]
    '''

    # initiate model
    FIT_G = fitting.LevMarLSQFitter()
    g_init = models.Gaussian1D(amplitude=np.nanmin(ccf) - np.nanmax(ccf), mean=rv_guess, stddev=stddev_guess) + models.Const1D(amplitude=np.nanmax(ccf))

    # First look for CCF peak around user-supplied stellar gamma velocity
    gaussian_fit_init = FIT_G(g_init, velocity_loop, ccf)

    # If the peak doesn't look right
    if gaussian_fit_init.amplitude_0 > 0 \
    or gaussian_fit_init.amplitude_0 > (0 - np.std(ccf)) \
    or gaussian_fit_init.stddev_0 < 0.4 \
    or gaussian_fit_init.mean_0 > np.max(velocity_loop) \
    or gaussian_fit_init.mean_0 < np.min(velocity_loop):
        print('No significant peak found in CCF')

    # do 'wide' fit to get rough estimate using parameters from initial fit
    g_wide = models.Gaussian1D(
             amplitude = gaussian_fit_init.amplitude_0,
             mean = gaussian_fit_init.mean_0,
             stddev = gaussian_fit_init.stddev_0) +\
             models.Const1D(amplitude=gaussian_fit_init.amplitude_1)
    rv_guess = gaussian_fit_init.mean_0

    # Fit smaller range around CCF peak
    i_fit = ((velocity_loop >= rv_guess - velocity_halfrange_to_fit) &
             (velocity_loop <= rv_guess + velocity_halfrange_to_fit))
    g_x = velocity_loop[i_fit]
    g_y = ccf[i_fit]
    gaussian_fit = FIT_G(g_wide, g_x, g_y)
    rv_mean = gaussian_fit.mean_0.value
    xi2 = sum((gaussian_fit(velocity_loop) - ccf)**2)
    return rv_mean, gaussian_fit, g_x, g_y, xi2

# basic fit for CCF (Gaussian)
def fit_gaussian_to_ccf_2(velocity_ccfloop, ccfout, rv_guess, velocity_halfrange_to_fit=10.0,stddev_guess=0.3):

    '''
    Fit a Gaussian to the cross-correlation function

    Note that median value of the CCF is subtracted before
    the fit, since astropy has trouble otherwise.

    Analyses of CCF absolute levels must be performed separately.

    Parameters
    ----------
    velocity_loop : array
        Input velocity step array for CCF [km/s]
    rv_guess : float
        Initial guess for RV
    velocity_halfrange_to_fit : float
        Full range of CCF to be fit (must be narrower than velocity_loop)
    stddev_guess : float
        guess for standard deviation of fit to CCF
    Returns
    -------
    gaussian_fit : object
        Final Gaussian fit (evaluated)
    gx, gy: arrays
        Narrowed CCF arrays for window that was actually used for final fit
    rv_mean: float
        Final RV measurement [km/s]
    '''
    # initiate guesses
    amplitude=np.nanmin(ccfout) - np.nanmax(ccfout)
    mean     =rv_guess
    stddev   =stddev_guess * GAUSSCONST
    offset   =np.nanmax(ccfout)

    p0 = [mean,stddev,amplitude,offset]

    # First look for CCF peak around user-supplied stellar gamma velocity
    bestfit    = opt.curve_fit(gaussian_fwhm,velocity_ccfloop,ccfout,p0=p0,#,bounds=bounds,\
                                method="trf")#,options={'maxiter' : 15000},tol=1e-20)

    p0 = bestfit[0]
    rv_guess = p0[0]

    # Fit smaller range around CCF peak
    i_fit = ((velocity_ccfloop >= rv_guess - velocity_halfrange_to_fit) &
             (velocity_ccfloop <= rv_guess + velocity_halfrange_to_fit))
    bestfit    = opt.curve_fit(gaussian_fwhm,velocity_ccfloop[i_fit],ccfout[i_fit],p0=p0,#,bounds=bounds,\
                                method="trf")#,options={'maxiter' : 15000},tol=1e-20)

    rv_fit = bestfit[0][0]
    xi2 = sum((gaussian_fwhm(velocity_ccfloop,*p0) - ccfout)**2)
    mean,stddev,amplitude,offset = p0
    if amplitude > 0  \
    or mean > np.max(velocity_ccfloop) \
    or mean < np.min(velocity_ccfloop) \
    or stddev < 0.1:
        print('No significant peak found in CCF')

    return rv_fit, velocity_ccfloop[i_fit],gaussian_fwhm(velocity_ccfloop[i_fit],*p0),xi2


def find_ccf_min(velocity_ccfloop, ccfout, rv_guess, velocity_halfrange_to_fit=10.0,stddev_guess=0.3):

    '''
    interpolate CCF and find minimum
    THIS DOESNT WORK WELL

    Note that median value of the CCF is subtracted before
    the fit, since astropy has trouble otherwise.

    Analyses of CCF absolute levels must be performed separately.

    Parameters
    ----------
    velocity_loop : array
        Input velocity step array for CCF [km/s]
    rv_guess : float
        Initial guess for RV
    velocity_halfrange_to_fit : float
        Full range of CCF to be fit (must be narrower than velocity_loop)
    stddev_guess : float
        guess for standard deviation of fit to CCF
    Returns
    -------
    gaussian_fit : object
        Final Gaussian fit (evaluated)
    gx, gy: arrays
        Narrowed CCF arrays for window that was actually used for final fit
    rv_mean: float
        Final RV measurement [km/s]
    '''
    # initiate guesses
    amplitude=np.nanmin(ccfout) - np.nanmax(ccfout)
    mean     =rv_guess
    stddev   =stddev_guess
    offset   =np.nanmax(ccfout)

    p0 = [mean,stddev,amplitude,offset]

    # First look for CCF peak around user-supplied stellar gamma velocity
    bestfit    = opt.curve_fit(gaussian_fwhm,velocity_ccfloop,ccfout,p0=p0,#,bounds=bounds,\
                                method="trf")#,options={'maxiter' : 15000},tol=1e-20)

    mean,stddev,amplitude,offset = bestfit[0]
    
    if amplitude > 0  \
    or mean > np.max(velocity_ccfloop) \
    or mean < np.min(velocity_ccfloop) \
    or stddev < 0.1:
        print('No significant peak found in CCF')

    p0 = bestfit[0]
    rv_guess = p0[0]

    # Fit smaller range around CCF peak
    i_fit = ((velocity_ccfloop >= rv_guess - velocity_halfrange_to_fit) &
             (velocity_ccfloop <= rv_guess + velocity_halfrange_to_fit))
    x, y = velocity_ccfloop[i_fit],ccfout[i_fit]

    tck = interpolate.splrep(x,y, k=3, s=0)
    newx = np.arange(np.min(x),np.max(x),0.01*1e-5) #0.01cm/s steps!
    newy = interpolate.splev(newx,tck,der=0,ext=1)
    rv_fit = newx[np.argmin(newy)]

    return rv_fit, newx, newy


def find_ccfderiv_zero(velocity_ccfloop, ccfout):

    '''
    take derivate of ccf and find 0 crossing point

    Note that median value of the CCF is subtracted before
    the fit, since astropy has trouble otherwise.

    Analyses of CCF absolute levels must be performed separately.

    Parameters
    ----------
    velocity_loop : array
        Input velocity step array for CCF [km/s]
    ccfout
    '''
    # initiate guesses
    deriv = np.gradient(ccfout)
    m,b = np.polyfit(velocity_ccfloop[1:-1],deriv[1:-1],deg=1)

    rv_fit = -1*b/m

    return rv_fit,deriv,velocity_ccfloop*m+b


# estimate photon-limited RV uncertainty of computed CCF
def ccf_error_calc(vel_arr, ccf, fit_wid=20., pix_wid_vel=0.6):
    '''
    Estimates photon-limited velocity uncertainty of cross-correlation
    function using methods in Boisse et al. 2010 (B10), Appendix A.2.

    Calculates weighted slope information of CCF and converts to
    approximate RV uncertainty based on photon noise alone.

    Modified slightly by SPH and JPN to include scale factor to account
    for relative velocity step size of CCF compared to native NEID pixels.

    Parameters
    ----------
    vel_arr : :obj:'arr' of :obj:`float`
        CCF velocity step array [km/s]

    ccf : :obj:'arr' of :obj:`float`
        CCF array

    fit_wid : :obj:`float`
        Width of CCF being fit for RVs [km/s]

    pix_wid_vel : :obj:`float`
        Velocity span per pixel [km/s/pixel]

    Returns
    -------
    sigma_ccf : :obj:`float`
        Estimated photon-limited uncertainty of RV measurement
        using specified ccf [km/s]

    S Halverson - JPL - 29-Sep-2019
    '''

    # get approximate CCF velocity step size
    ccf_vel_step = np.mean(np.diff(vel_arr)) # km/s

    # scaling factor used to compensate for the difference between
    # width of CCF velocity step in units of native spectrum pixels
    # used to correct for CCF 'oversampling'
    n_scale_pix = ccf_vel_step / pix_wid_vel # number of NEID spectral pixels per ccf step

    # isolate only velocity steps in the CCF that are used for fitting
    inds_fit = np.where((vel_arr >= ((-1.) * fit_wid / 2.)) & (vel_arr <= (fit_wid / 2.)))
    vels_fit = vel_arr[inds_fit]
    ccf_fit = ccf[inds_fit]

    # noise of each CCF point (assumed to be photon noise only)
    noise_ccf = (ccf_fit) ** 0.5

    # calculate slopes
    deriv_ccf = np.gradient(ccf_fit, vels_fit)

    #weight them by the noise
    weighted_slopes = (deriv_ccf) ** 2. / (noise_ccf) ** 2.

    # numerator of equation A.2 in B10
    top = (np.sum(weighted_slopes)) ** 0.5

    # demonimator of equation A.2 in B10
    bottom = (np.sum(ccf_fit)) ** 0.5

    # calculate Q-factor of CCF
    qccf = (top / bottom) * (n_scale_pix ** 0.5)

    # calculate final ccf uncertainty
    sigma_ccf = 1. / (qccf * ((np.sum(ccf_fit)) ** 0.5)) # km/s

    return sigma_ccf