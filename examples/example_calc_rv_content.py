import numpy as np
from astropy.modeling import models, fitting
from scipy import signal, interpolate

import pandas as pd
import matplotlib.pylab as plt
import sys,os,glob,random

plt.ion()
plt.rcParams['font.size'] = '12'

sys.path.append('/Users/ashbake/Documents/Research/Projects/HISPEC/RV_simulations/rvsim/')
import ccf,resample
from source import load_phoenix_spectrum, shift_spectrum, mod_spec
from mask import load_phoenix_mask
from spectrograph import load_order_bounds

from astropy.io import fits
import pandas as pd


SPEEDOFLIGHT = 2.998e8 # m/s
GAUSSCONST = (2. * (2. * np.log(2))**0.5)

plt.ion()


def plot_rv_err_lfc(snr_arr,rv_arr,ax=None,label=''):
	"""
	plot RV error

	inputs:
	------
	snr_arr - snrs
	rv_arr  - rv precisions
	ax      - figure axis (optional, default: None)
	label   - label for figure (option, default '')
	"""
	if ax == None:
		fig, ax = plt.subplots(1,figsize=(7,5))
	plt.subplots_adjust(bottom=0.15,left=0.15,right=0.85,top=0.85)

	ax.loglog(snr_arr,rv_arr,label=label)
	ax.set_ylabel('$\sigma_{RV}$ [m/s]')
	ax.set_xlabel('SNR')

	ax.grid(True)
	ax.set_title('RV Error vs Cal. SNR')
	plt.legend()

	return ax


def get_rv_content(v,s,n):
	"""
	given spectrum and its noise, get the RV error content

	inputs
	------
	v - wavelength [nm]
	s - spectrum [photons]
	n - noise (same units as s)
	"""
	flux_interp = interpolate.InterpolatedUnivariateSpline(v,s, k=1)
	dflux = flux_interp.derivative()
	spec_deriv = dflux(v)
	sigma_ord = np.abs(n) #np.abs(s) ** 0.5 # np.abs(n)
	sigma_ord[np.where(sigma_ord ==0)] = 1e10
	all_w = (v ** 2.) * (spec_deriv ** 2.) / sigma_ord ** 2. # include read noise and dark here!!
	
	return all_w



def get_rv_err_ord(dfreq,snr,rn,wavbounds,R=100000,ploton=False):
	"""
	Creates a fake LFC or Etalon spectrum given line frequency then
	Calculates the RV precision content of that spectrum at some snr

	inputs
	------
	dfreq - line spacing of etalon/lfc spectrum
	snr   - signal to noise ratio of spectrum. assumes photon noise dominated.
	rn    - read noise in photoelectrons to assume
	wavbounds - wavelength limits of order in question in form of tuple e.g. [890,900] [nm]
	R     - resolving power of spectrum 
	ploton - default False. If true, will plot the simulated spectrum

	output:
	------
	dv_order: the rv precision for the order given

	"""
	dlam =np.mean(wavbounds)/R/3
	v = np.arange(wavbounds[0],wavbounds[1],dlam)

	flo,fhi    = SPEEDOFLIGHT/(np.max(v)), SPEEDOFLIGHT/(np.min(v)) # GHz
	line_freqs = np.arange(flo,fhi,dfreq)
	line_wvls  = SPEEDOFLIGHT/line_freqs 
	weights    = np.ones_like(line_wvls)
	fwhms      = np.ones_like(line_wvls) * dlam * 3
	spectrum   = 1 - ccf.spec_make(v, weights, line_wvls, fwhms)
	s = spectrum * snr**2
	sigma = np.sqrt(s + 3 * rn**2) 

	# calc RV precision for that spectrum
	w_ord		= get_rv_content(v,s,sigma)
	dv_order    = SPEEDOFLIGHT / (np.nansum(w_ord[1:-1])**0.5)
	
	if ploton:
		plt.figure(figsize=(7,4))
		plt.plot(v,s/sigma)
		plt.xlabel('Wavelength (nm)')
		plt.ylabel('SNR')
		plt.title('LFC (freq=%sGHz)'%dfreq)
		plt.subplots_adjust(bottom=0.15)
		plt.savefig('LFC_spec_%sGHz.png'%dfreq)
		#plt.xlim(2100,2101.5)
		#plt.savefig('LFC_spec_%sGHz_zoom.png'%dfreq)

	return dv_order

if __name__=='__main__':
	# change to use spec_rv_noise_calc in ccd_tools.py
	orders = np.arange(59,68)#149)
	specname='HK'
	snrs = np.array([1,5,10,50,100,200,500,1000])
	
	dv_all = np.zeros_like(snrs,dtype=float)
	for i,snr in enumerate(snrs):
		dv_ords = np.zeros_like(orders,dtype=float)
		for j,order in enumerate(orders):
			# order bounds and make wavelength grid
			wavbounds=load_order_bounds(spec='HK',order=order)

			# LFC
			# make fake spectrum LFC/etalon	
			dv_ords[j]  = get_rv_err_ord(40,snr,3,wavbounds,R=50000,ploton=False)
		dv_all[i] = 1/np.sqrt(np.sum(1/dv_ords)**2)



	ax = plot_rv_err_lfc(snrs,dv_all,ax=None,label='HPF H Band')

	orders = np.arange(59,68)#149)
	specname='HK'
	snrs = np.array([1,5,10,50,100,200,500,1000])
	
	dv_all = np.zeros_like(snrs,dtype=float)
	for i,snr in enumerate(snrs):
		dv_ords = np.zeros_like(orders,dtype=float)
		for j,order in enumerate(orders):
			# order bounds and make wavelength grid
			wavbounds=load_order_bounds(spec='HK',order=order)

			# LFC
			# make fake spectrum LFC/etalon	
			dv_ords[j]  = get_rv_err_ord(16,snr,3,wavbounds,R=100000,ploton=False)
		dv_all[i] = 1/np.sqrt(np.sum(1/dv_ords)**2)


	plot_rv_err_lfc(snrs,dv_all,ax=ax,label='HISPEC H Band LFC')


	orders = np.arange(59,68)#149)
	specname='HK'
	snrs = np.array([1,5,10,50,100,200,500,1000])
	
	dv_all = np.zeros_like(snrs,dtype=float)
	for i,snr in enumerate(snrs):
		dv_ords = np.zeros_like(orders,dtype=float)
		for j,order in enumerate(orders):
			# order bounds and make wavelength grid
			wavbounds=load_order_bounds(spec='HK',order=order)

			# LFC
			# make fake spectrum LFC/etalon	
			dv_ords[j]  = get_rv_err_ord(30,snr,3,wavbounds,R=100000,ploton=False)
		dv_all[i] = 1/np.sqrt(np.sum(1/dv_ords)**2)


	
	plot_rv_err_lfc(snrs,dv_all,ax=ax,label='HISPEC H Band Etalon')



