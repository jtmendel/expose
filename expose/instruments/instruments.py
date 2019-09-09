from __future__ import print_function
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from scipy.interpolate import interp1d

import numpy as np
import os
import glob
import sys

from ..utils.interpolation import interpolate_os
from ..utils.smoothing import smooth


__all__ = ["instrument", "MUSE", "MAVIS"]


class instrument:
    """
    Generic instrument class with some generic instrument routines.  
    """
    def __init__(self):
        pass
    
    def _resample(self, x_output, x_input, y_input):
        rs = interpolate_os(x_output, x_input, y_input)
        rs[0] = np.interp(x_output[0], x_input, y_input)
        rs[-1] = np.interp(x_output[-1], x_input, y_input)
        return rs
 
    def _EE_gaussian(self, seeing, binning=1):
        sigma_pix = seeing / self.pix_scale / 2.355
        
        #assume seeing at 500nm a la ESO ETC
        #estimate flux fraction given binning
        profile_int = Gaussian2DKernel(sigma_pix, x_size=binning, 
                                       y_size=binning).array.sum()
        #compute seeing variation over the spectral range
        sigma_pix_array = sigma_pix * (self.inst_wavelength/.500)**(-1/5)
                
        ee_array = profile_int * sigma_pix**2 / sigma_pix_array**2
        return ee_array    

    def observe(self, source, telescope, sky=None, dit=3600., 
                ndit=1, seeing=1., binning=1):
        
        #generate source
        source_wave, source_phot = source()
        
        #convolve source to instrument resolution (if higher) or else leave be
        match_res_source = np.interp(source_wave, self.inst_wavelength, self.res_pix,
                                     left=self.res_pix[0], right=self.res_pix[-1])
        offset_res_source = np.sqrt(np.clip((match_res_source*self.step/source.step)**2 - 
                                            source.res_pix**2, 1e-10,None))
        conv_source = smooth(source_phot, offset_res_source)
        
        #resample onto outputpixel grid
        source_resampled = self._resample(self.inst_wavelength, source_wave, conv_source)
        
        #if a sky object is also supplied, convolve it to match
        if sky is not None:
            sky_wave, sky_emm, sky_trans = sky()
            
            match_res_sky = np.interp(sky_wave, self.inst_wavelength, self.res_pix,
                                     left=self.res_pix[0], right=self.res_pix[-1])
            offset_res_sky = np.sqrt(np.clip((match_res_sky*self.step/sky.step)**2 - 
                                             sky.res_pix**2, 1e-10, None))
            conv_emm = smooth(sky_emm, offset_res_sky)
            conv_trans = smooth(sky_trans, offset_res_sky)
                        
            #resample onto output grid
            sky_emm_resampled = self._resample(self.inst_wavelength, sky_wave, conv_emm)
            sky_trans_resampled = np.clip(self._resample(self.inst_wavelength, 
                                                         sky_wave, conv_trans),0,1)
            
        else:
            sky_trans_resampled = np.ones(len(self.inst_wavelength))
            sky_emm_resampled = np.zeros(len(self.inst_wavelength))
            
        #total source spectrum
        source_obs = np.copy(source_resampled)*sky_trans_resampled*dit*\
                     self.tpt*self.step*telescope.area*self._ee(seeing, binning=binning) #total area
            
        sky_obs = np.copy(sky_emm_resampled)*sky_trans_resampled*dit*self.tpt*\
                  self.step*telescope.area*self.pix_scale**2 * binning**2 #total area
            
        #total noise calculation
        noise = source_obs + sky_obs + binning**2 * self.rn**2 + self.dark*binning**2 * dit
            
        sn = np.sqrt(ndit)*source_obs / np.sqrt(noise)
        sn_A = np.sqrt(1./1e4/self.step) * np.copy(sn)
        return self.inst_wavelength, sn
   

class MUSE(instrument):
    """ 
    A MUSE-like instrument.  
    
    Assumes a simple gaussian model for the PSF and simple
    linear model for spectral resolution.
    """

    def __init__(self):
        #initialize the model base
        instrument.__init__(self)

        #wavelength business
        self.step = 1.25 / 1e4 #microns
        self.wmin, self.wmax = 4800./1e4, 9300./1e4
        self.inst_wavelength = np.arange(self.wmin, self.wmax, self.step)

        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['expose'].__file__), 'data')
        
        #for the moment adopt MUSE tabulated throughput, modified by MAVIS AO sim throughput
        tx, ty = [], []
        with open(os.path.join(bfile_dir, 'muse_throughput.txt'),'r') as file:
            for line in file:
                temp = line.strip().split(None)
                tx.append(float(temp[0]))
                ty.append(float(temp[1])/100.)
        self.tpt = np.interp(self.inst_wavelength, np.array(tx)/1e3, ty, 
                             left=ty[0], right=ty[-1])        

        #detector parameters
        self.rn = 3.0 #e-/pixel/dit
        self.dark = 3.0/3600. #e-/pixel/s
        
        #pixel properties
        self.pix_scale = 0.2 #arcsec per pixel
        
        #resolution/LSF information
        self.res_wave = np.array([0.480, 0.930])
        self.res_power = np.array([1770,3590])
        self.res_power_interp = np.interp(self.inst_wavelength, self.res_wave, self.res_power, 
                             left=self.res_power[0], right=self.res_power[1])
        self.res_pix = self.inst_wavelength / self.res_power_interp / self.step / 2.355 #sigma, in pixels
        
        #ensquared energy
        self._ee = self._EE_gaussian


class MAVIS(instrument):
    """
    A MAVIS-like instrument.

    Assumes that the general properties of the MAVIS spectrograph
    are MUSE-like, but with an added throughput hit from the AO system
    and more elaborate PSF model.
    """

    def __init__(self, R=7000, pix_scale=0.007):
        #initialize the model base
        instrument.__init__(self)

        #fixed for the moment by the EE profiles
        self.pix_scale = pix_scale
        
        #detector parameters (adopted from MUSE)
        self.rn = 3.0 #e-/pixel/dit
        self.dark = 3.0/3600. #e-/pixel/s
        
        #generate wavelength array based on critically sampling R
        self.wmin, self.wmax = 3700./1e4, 9500./1e4

        self.res_power = R
        self.step = 0.7/self.res_power/2. #microns
        self.inst_wavelength = np.arange(self.wmin, self.wmax, self.step)
        self.res_pix = self.inst_wavelength / self.res_power / self.step / 2.355 #sigma, in pixels
        
        
        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['expose'].__file__), 'data')

        
        #for the moment adopt MUSE tabulated throughput, modified by MAVIS AO sim throughput
        tx, ty = [], []
        with open(os.path.join(bfile_dir, 'muse_throughput.txt'),'r') as file:
            for line in file:
                temp = line.strip().split(None)
                tx.append(float(temp[0]))
                ty.append(float(temp[1])/100.)
        self.tpt = np.interp(self.inst_wavelength, np.array(tx)/1e3, ty, 
                             left=ty[0], right=ty[-1])        

        tx, ty = [], []
        with open(os.path.join(bfile_dir, 'mavis_AOM_throughput.txt'),'r') as file:
            for line in file:
                if line[0] != '#':
                    temp = line.strip().split(None)
                    tx.append(float(temp[0])) #already um
                    ty.append(float(temp[2])) #using the 4OAP profile
        self.ao_tpt = np.interp(self.inst_wavelength, np.array(tx), ty,
                                left=ty[0], right=ty[-1])
        self.tpt *= self.ao_tpt
        
        
        #pre-load EE profiles
        ee_files = glob.glob(os.path.join(bfile_dir, '*EEProfile.dat'))
        wave = []
        ee_interp = []
        for ii, ee_file in enumerate(ee_files):
            twave = os.path.split(ee_file)[1].split('_')[3][:-2]
            wave.append(float(twave)/1e3)
            with open(ee_file, 'r') as file:
                tr, tee = [], []
                for line in file:
                    temp = line.strip().split(',')
                    tr.append(float(temp[0])/self.pix_scale) #in pixels
                    tee.append(float(temp[1]))
                ee_interp.append(interp1d(tr, tee, bounds_error=False, fill_value='extrapolate'))
        self._ee_profile_wave = np.array(wave)
        self._ee_profile_interp = ee_interp
        
        self._ee = self._EE_lookup
        
    def _EE_lookup(self, seeing, binning=1):
        """
        A quick and dirty lookup/interpolation function to generate ensquared 
        energy profiles based on simulations of the MAVIS PSF.
        """
        #Seeing isn't relevant for MAVIS at the moment, so it only depends on the binning
        iwave = np.argsort(self._ee_profile_wave)
        
        wave_out = np.zeros(len(iwave), dtype=np.float)
        ee_out = np.zeros(len(iwave), dtype=np.float)
        for ii, idx in enumerate(iwave): #sorted arguments?
            wave_out[ii] = self._ee_profile_wave[idx]
            ee_out[ii] = self._ee_profile_interp[idx](binning)
   
        ee_interped = interp1d(wave_out, ee_out, fill_value='extrapolate')(self.inst_wavelength)
        return ee_interped
    
