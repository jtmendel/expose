from __future__ import print_function
from astropy.io import fits, ascii as asc
from astropy.convolution import Gaussian2DKernel
from scipy.interpolate import interp1d

import numpy as np
import os
import glob
import sys
import fsps

from ..utils.smoothing import smooth


__all__ = ["ImagingInstrument", "DREAMS_Imager", "MAVIS_Imager"]


class ImagingInstrument:
    """
    Generic instrument class with some generic instrument routines.  
    """
    def __init__(self):
        self.small_num = 1e-70
        self.source_obs = None
        self.sky_obs = None
   
        self.transmission = None
        self.wavelength = None
        self.pivot = None

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

    def _set_filter(self, filt):
        #fetch filter transmission curve from FSPS
        #this sets the wavelength grid, so no rebinning needed

        #lookup for filter number given name
        fsps_filts = fsps.list_filters()
        filt_lookup = dict(zip(fsps_filts, range(1,len(fsps_filts)+1)))

        #reference in case given a spitzer or mips filter...probably not an issue right now.
        mips_dict = {90:23.68*1e4, 91:71.42*1e4, 92:155.9*1e4}
        spitzer_dict = {53:3.550*1e4, 54:4.493*1e4, 55:5.731*1e4, 56:7.872*1e4}
        
        #pull information for this filter
        fobj = fsps.get_filter(filt)
        filter_num = filt_lookup[filt]
        
        fwl, ftrans = fobj.transmission

        ftrans = np.maximum(ftrans, 0.)
        trans_interp = np.asarray(np.interp(self.inst_wavelength, fwl/1e4, 
                                  ftrans, left=0., right=0.), dtype=np.float)

 
        #normalize transmission
        ttrans = np.trapz(np.copy(trans_interp)/self.inst_wavelength, self.inst_wavelength)
        if ttrans < self.small_num: ttrans = 1.
        ntrans = np.maximum(trans_interp / ttrans, 0.0)
        
        if filter_num in mips_dict:
            td = np.trapz(((self.inst_wavelength/mips_dict[filter_num])**(-2.))*ntrans/self.inst_wavelength, self.inst_wavelength)
            ntrans = ntrans/max(1e-70,td)

        if filter_num in spitzer_dict:
            td = np.trapz(((self.inst_wavelength/spitzer_dict[filter_num])**(-1.0))*ntrans/self.inst_wavelength, self.inst_wavelength)
            ntrans = ntrans/max(1e-70,td)

        #stupid, but re-normalize to peak of 1 (since all other throughput terms 
        #are included in the instrument throughput

        self.trans_norm = np.copy(ntrans)/ntrans.max()
        self.transmission = ntrans
        self.pivot = fobj.lambda_eff
        return 

              

    def observe(self, source, telescope, sky=None, dit=3600., 
                ndit=None, sn=None, seeing=1., binning=1, band='v'):
     
        #set filter
        self._set_filter(band)

        #generate source
        source_wave, source_phot = source()

        #resample onto outputpixel grid
        source_resampled = np.interp(self.inst_wavelength, source_wave, source_phot)
        
        #if a sky object is also supplied, convolve it to SOURCE
        if sky is not None:
            sky_wave, sky_emm, sky_trans = sky()
            
            match_res_sky = np.interp(sky_wave, source_wave, source.res_pix,
                                     left=source.res_pix[0], right=source.res_pix[-1])
            offset_res_sky = np.sqrt(np.clip((match_res_sky*source.step/sky.step)**2 - 
                                             sky.res_pix**2, 1e-10, None))
            conv_emm = smooth(sky_emm, offset_res_sky)
            conv_trans = smooth(sky_trans, offset_res_sky)
                        
            #resample onto output grid
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

        #### this is all without the filter because of wavelength dependence
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
            
        #fold in transmission for total counts 
        self.source_obs = source_obs*self.trans_norm
        self.sky_obs = sky_obs*self.trans_norm

        #total noise calculation
        self.noise = self.source_obs.sum() + self.sky_obs.sum() + self.obs_area*(self.rn**2 + self.dark*dit) #per dit
        
        if sn is not None and ndit is None: #provided S/N, work out ndit to reach S/N 5
            ndit = np.int(np.ceil(np.sqrt(self.noise)*sn/self.source_obs.sum()))
            print("NDIT={2} to reach S/N={0} with DIT={1}".format(sn, dit, ndit))
        elif sn is None and ndit is not None:
            sn = np.sqrt(ndit)*self.source_obs.sum() / np.sqrt(self.noise)
            print("S/N={0} at with NDIT={1} and DIT={2}".format(sn, ndit, dit))


        sn = np.sqrt(ndit)*self.source_obs.sum() / np.sqrt(self.noise)
        return self.pivot, sn
   

class DREAMS_Imager(ImagingInstrument):
    """ 
    A DREAMS-like instrument. 
    
    Assumes a simple gaussian model for the PSF.
    """

    def __init__(self, pix_scale=3.59):
        #initialize the model base
        ImagingInstrument.__init__(self)

        #wavelength business
        self.step = 1. / 1e4 #microns
        self.wmin, self.wmax = 9700./1e4, 17800./1e4
        self.inst_wavelength = np.arange(self.wmin, self.wmax, self.step)

        #set the pixel scale
        self.pix_scale = pix_scale
 
        #get path for bundled package files
        bfile_dir = os.path.join(os.path.dirname(sys.modules['expose'].__file__), 'data')
        
        #detector parameters
        self.rn = 35.0 #e-/pixel/dit
        self.dark = 300. #e-/pixel/s
       
        #Princeton detector stats
        data = np.array(asc.read(os.path.join(bfile_dir, '1280scicam_QE.csv')))
        twave = np.array(data['col1'])
        tqe = np.array(data['col2'])
        p_qe = interp1d(twave, tqe, fill_value='extrapolate')(self.inst_wavelength)

        #assume high reflectivity from gold coatings coatings
        self.tel_tpt = 0.96**2 * p_qe * 0.8 #80 % fudge
        self.tpt = self.tel_tpt * 0.98**2 * 0.9

        #simple Gaussian EE
        self._ee = self._EE_gaussian


class MAVIS_Imager(ImagingInstrument):
    """
    A MAVIS-like instrument.

    Assumes that the general properties of the MAVIS spectrograph
    are MUSE-like, but with an added throughput hit from the AO system
    and more elaborate PSF model.
    """

    def __init__(self, pix_scale=0.007, jitter=5):
        #check for reasonable jitter values
        if jitter not in [5,10,20,30,40]:
            raise ValueError('Input jitter must be one of 5, 10, 20, 30, or 40 (mas)')

        #initialize the model base
        ImagingInstrument.__init__(self)

        #set the pixel scale
        self.pix_scale = pix_scale
       
        #wavelength business
        self.step = 1. / 1e4 #microns
        self.wmin, self.wmax = 3700./1e4, 10100./1e4
        self.inst_wavelength = np.arange(self.wmin, self.wmax, self.step)

        #detector parameters (adopted from MUSE)
        self.rn = 3.0 #e-/pixel/dit
        self.dark = 3.0/3600. #e-/pixel/s

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

        #compute the combined throughput - including filter transmission
        self.tel_tpt = m1_reflect*m2_reflect*m3_reflect*e2v_qe
        self.tpt = self.tel_tpt*0.98**2

        #patch in low throughput at the notch
        notch = (self.inst_wavelength > 0.580) & (self.inst_wavelength < 0.600)
        self.tpt[notch] = 1e-5

        #fold in AOM throughput
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
 
