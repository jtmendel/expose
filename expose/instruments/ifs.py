from __future__ import print_function
from astropy.io import fits, ascii as asc
from astropy.convolution import Gaussian2DKernel
from scipy.interpolate import interp1d

import numpy as np
import os
import glob
import sys

from ..utils.interpolation import interpolate_os
from ..utils.smoothing import smooth


__all__ = ["IFSInstrument", "MUSE", "MAVIS_IFS"]


class IFSInstrument:
    """
    Generic instrument class with some generic instrument routines.  
    """
    def __init__(self):
        self.source_obs = None
        self.sky_obs = None
    

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
        return ee_array, binning**2    


    def observe(self, source, telescope, sky=None, dit=3600., 
                ndit=None, sn=None, seeing=1., binning=1, ref_wl=0.7):
        
        #generate source
        source_wave, source_phot = source(wavelength=self.inst_wavelength, resolution=self.res_pix)
        
        #convolve source to instrument resolution (if higher) or else leave be
        match_res_source = np.interp(source_wave, self.inst_wavelength, self.res_pix,
                                     left=self.res_pix[0], right=self.res_pix[-1]) #instrument resolution on source wavelength grid
        offset_res_source = np.sqrt(np.clip((match_res_source*self.step/source.red_step)**2 - 
                                            source.res_pix**2, 1e-30,None))
        conv_source = smooth(source_phot, offset_res_source)
       

        #resample onto outputpixel grid
        #source_resampled = self._resample(self.inst_wavelength, source_wave, conv_source)
        source_resampled = np.interp(self.inst_wavelength, source_wave, conv_source)
        
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
            #sky_emm_resampled = self._resample(self.inst_wavelength, sky_wave, conv_emm)
            #sky_trans_resampled = np.clip(self._resample(self.inst_wavelength, 
            #                                             sky_wave, conv_trans),0,1)
            sky_emm_resampled = np.interp(self.inst_wavelength, sky_wave, conv_emm)
            sky_trans_resampled = np.clip(np.interp(self.inst_wavelength, 
                                                         sky_wave, conv_trans),0,1)
            
        else:
            sky_trans_resampled = np.ones(len(self.inst_wavelength))
            sky_emm_resampled = np.zeros(len(self.inst_wavelength))

        #store transmission spectrum
        self.sky_trans = np.copy(sky_trans_resampled)
        
        #get ensquared energy and area in pixels
        self.obs_ee, self.obs_area = self._ee(seeing, binning=binning)


        #total source spectrum
        if source.norm_sb:
            self.cfact = sky_trans_resampled*dit*self.tpt*self.step*\
                         telescope.area*self.pix_scale**2
        else:
            self.cfact = sky_trans_resampled*dit*self.tpt*self.step*\
                         telescope.area*self.obs_ee

        source_obs = np.copy(source_resampled)*self.cfact

        #sky is always done correctly-ish.
        sky_obs = np.copy(sky_emm_resampled)*sky_trans_resampled*dit*self.tpt*\
                  self.step*telescope.area*self.pix_scale**2 * self.obs_area #total area
            
        
        self.source_obs = np.copy(source_obs)
        self.sky_obs = np.copy(sky_obs)

        #total noise calculation
        self.noise = source_obs + sky_obs + self.obs_area*(self.rn**2 + self.dark*dit) #per dit
        
        if sn is not None and ndit is None: #provided S/N, work out ndit to reach S/N 5
            ndit = np.int(np.ceil(np.sqrt(self.noise)*sn/source_obs))
            ndit_i = np.interp(ref_wl, self.inst_wavelength, ndit)
            print("NDIT={2} to reach S/N={0} with DIT={1} at {3}um".format(sn, dit, ndit_i, ref_wl))
        elif sn is None and ndit is not None:
            sn = np.sqrt(ndit)*source_obs / np.sqrt(self.noise)
            sn_i = np.interp(ref_wl, self.inst_wavelength, sn) 
            print("S/N={0} at {3}um with NDIT={1} and DIT={2}".format(sn_i, ndit, dit, ref_wl))


        sn = np.sqrt(ndit)*source_obs / np.sqrt(self.noise)
        sn_A = np.sqrt(1./1e4/self.step) * np.copy(sn)
        return self.inst_wavelength, sn
   

class MUSE(IFSInstrument):
    """ 
    A MUSE-like instrument.  
    
    Assumes a simple gaussian model for the PSF and simple
    linear model for spectral resolution.
    """

    def __init__(self, mode='WFM'):
        #initialize the model base
        IFSInstrument.__init__(self)

        #wavelength business
        self.step = 1.25 / 1e4 #microns
        self.wmin, self.wmax = 4800./1e4, 9300./1e4
        self.inst_wavelength = np.arange(self.wmin, self.wmax, self.step)

        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['expose'].__file__), 'data/muse')
        
        #detector parameters
        self.rn = 3.0 #e-/pixel/dit
        self.dark = 3.0/3600. #e-/pixel/s
        
        #resolution/LSF information
        self.res_wave = np.array([0.480, 0.930])
        self.res_power = np.array([1770,3590])
        self.res_power_interp = np.interp(self.inst_wavelength, self.res_wave, self.res_power, 
                             left=self.res_power[0], right=self.res_power[1])
        self.res_pix = self.inst_wavelength / self.res_power_interp / self.step / 2.355 #sigma, in pixels
   
        #mode-dependent configuration
        if mode == 'WFM':
            self.pix_scale = 0.2 #arcsec per pixel
            self._ee = self._EE_gaussian
            tpt_file = 'muse_throughput_WFM.txt'
        if mode == 'NFM':
            self.pix_scale = 0.025 #arcsec per pixel
            self._ee = self._EE_nfm
            tpt_file = 'muse_throughput_NFM.txt'
            
        #read in tabulated throughput
        tx, ty = [], []
        with open(os.path.join(bfile_dir, tpt_file), 'r') as file:
            for line in file:
                temp = line.strip().split(None)
                tx.append(float(temp[0]))
                ty.append(float(temp[1])/100.)
        self.tpt = np.interp(self.inst_wavelength, np.array(tx)/1e3, ty, 
                             left=ty[0], right=ty[-1])        
       

    def _EE_nfm(self, seeing, binning=1):
        #kludgy handling of the NFM PSF
        ee_wave = np.array([0.55,0.75,0.85])
        ee_ref = np.array([[0.000,0.000,0.000],
                           [0.012,0.032,0.045],
                           [0.050,0.104,0.139],
                           [0.093,0.180,0.227],
                           [0.134,0.243,0.291],
                           [0.171,0.290,0.335]])
        ee_pixels = np.arange(6)
        ee_interp = interp1d(ee_pixels, ee_ref, fill_value='extrapolate', axis=0)(binning)

        ee_out = interp1d(ee_wave, ee_interp, fill_value=(ee_interp[0], ee_interp[-1]), 
                          bounds_error=False)(self.inst_wavelength)
        
        return ee_out, binning**2


class MAVIS_IFS(IFSInstrument):
    """
    A MAVIS-like instrument.

    Assumes that the general properties of the MAVIS spectrograph
    are MUSE-like, but with an added throughput hit from the AO system
    and more elaborate PSF model.
    """

    def __init__(self, mode=None, pix_scale=0.007, jitter=5):
        #check for reasonable jitter values
        if jitter not in [5,10,20,30,40]:
            raise ValueError('Input jitter must be one of 5, 10, 20, 30, or 40 (mas)')

        if mode not in ['LR-blue','LR-red','HR-blue','HR-red']:
            raise ValueError('Invalid grating setup.  Must be one of LR-blue, LR-red, HR-blue, or HR-red.')

        #initialize the model base
        IFSInstrument.__init__(self)

        #set the pixel scale
        self.pix_scale = pix_scale
        
        #detector parameters (adopted from MUSE)
        self.rn = 3.0 #e-/pixel/dit
        self.dark = 3.0/3600. #e-/pixel/s
        self.npix_det = 9000 #adjust?
        self.live_pix_det = 7500

        #grating parameters
        self.grating_dict = {
                'LR-blue': (3700, 3450),
                'LR-red': (5150, 3450),
                'HR-blue': (4250, 12800),
                'HR-red': (6300, 9600),
                }
        self.wmin, self.wmax = 3700./1e4, 10070./1e4

        #pull correct grating parameters
        grating_wmin, grating_rmin = self.grating_dict[mode]

        #generate full wavelength and resolution arrays
        self.dlam = grating_wmin / grating_rmin / 1e4
        self.step = self.dlam / 2.3

        self.inst_wavelength = np.arange(self.live_pix_det)*self.step + grating_wmin / 1e4
        self.wmin_eff, self.wmax_eff = self.inst_wavelength[0], self.inst_wavelength[-1]

        self.res_power_interp = self.inst_wavelength / 2.3 / self.step

        self.res_pix = self.inst_wavelength / self.res_power_interp / self.step / 2.355

        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['expose'].__file__), 'data')

        #E2V
        data = np.array(asc.read(os.path.join(bfile_dir, 'E2V_QE.csv')))
        twave = np.array(data['col1'])/1e3
        tqe = np.array(data['col2'])/100. 
        e2v_qe = interp1d(twave, tqe, fill_value='extrapolate')(self.inst_wavelength)
        
        #M1
        data = np.array(asc.read(os.path.join(bfile_dir, 'UT4_M1_reflect.csv')))
        twave = np.array(data['col1'])/1e3
        tref = np.array(data['col2'])/100.
        m1_reflect = interp1d(twave, tref, fill_value='extrapolate')(self.inst_wavelength)
        
        #M2
        data = np.array(asc.read(os.path.join(bfile_dir, 'UT4_M2_reflect.csv')))
        twave = np.array(data['col1'])/1e3
        tref = np.array(data['col2'])/100.
        m2_reflect = interp1d(twave, tref, fill_value='extrapolate')(self.inst_wavelength)
        
        #M3
        data = np.array(asc.read(os.path.join(bfile_dir, 'UT4_M3_reflect.csv')))
        twave = np.array(data['col1'])/1e3
        tref = np.array(data['col2'])/100.
        m3_reflect = interp1d(twave, tref, fill_value='extrapolate')(self.inst_wavelength)

        #VPH
        data = np.array(asc.read(os.path.join(bfile_dir, 'grating_eff.csv')))
        twave = np.array(data['col1'])/1e3
        teff = np.array(data['col2'])/100.

        #re-normalize the VPH coverage to match the necessary instrument wavelength,
        # i.e. assume that this is a generic description of the grating efficiency
        tnorm = (np.copy(twave)-twave.min())*self.inst_wavelength.ptp()/twave.ptp() + self.inst_wavelength.min()
        vph_eff = interp1d(tnorm, teff, fill_value='extrapolate')(self.inst_wavelength)

        #compute the combined throughput
        self.tel_tpt = m1_reflect*m2_reflect*m3_reflect*e2v_qe
        self.tpt = self.tel_tpt*vph_eff*0.60

        #patch in low throughput at the notch
        notch = (self.inst_wavelength > 0.580) & (self.inst_wavelength < 0.600)
        self.tpt[notch] = 1e-5

        #UPDATE AOM THROUGHPUT ESTIMATE
        data = np.array(asc.read(os.path.join(bfile_dir, 'mavis/mavis_AOM_throughput.csv')))
        twave = np.array(data['col1'])
        ttpt = np.array(data['col2'])

        self.ao_tpt = np.interp(self.inst_wavelength, np.array(twave), np.array(ttpt),
                                left=ttpt[0], right=ttpt[-1])
        self.tpt *= self.ao_tpt

        
        #pre-load EE profiles
        ee_files = glob.glob(os.path.join(bfile_dir, 'mavis/PSF_{0}mas*EEProfile.dat'.format(jitter)))
        wave = []
        ee_interp = []
        for ii, ee_file in enumerate(ee_files):
            twave = os.path.split(ee_file)[1].split('_')[2][:-2]
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
            ee_out[ii] = self._ee_profile_interp[idx](binning/2.)
   
        ee_interped = interp1d(wave_out, ee_out, fill_value='extrapolate')(self.inst_wavelength)
        return ee_interped, binning**2
    
