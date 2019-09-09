from __future__ import print_function
import numpy as np
import skycalc_cli
import json
import os

from astropy.io import fits

__all__ = ["sky_source"]

class sky_source():
    """
    Class to handle generating a sky spectrum using the ESO advanced 
    sky model.  Sky can be computed for different FLI, airmass, and PWV.
    
    Output sky radiance is in Photons/s/m^2/um/arcsec^2.
    """
    def __init__(self, fli=0.5, airmass=1.2, pwv=5, 
                resolution=20000):
        
        self.res = resolution
        self.step = 1./self.res/2. #set step sized based on resolution at 1um
        
        self.paramDict = {'airmass': 1.0,
                          'pwv_mode':  'pwv',
                          'season': 0,
                          'time': 0,
                          'pwv': 3.5,
                          'msolflux': 130.0,
                          'incl_moon':'Y',
                          'moon_sun_sep': 90.0,
                          'moon_target_sep': 45.0,
                          'moon_alt': 45.0,
                          'moon_earth_dist': 1.0,
                          'incl_starlight': 'Y',
                          'incl_zodiacal': 'Y',
                          'ecl_lon': 135.0,
                          'ecl_lat': 90.0,
                          'incl_loweratm': 'Y',
                          'incl_upperatm': 'Y',
                          'incl_airglow': 'Y',
                          'incl_therm': 'N',
                          'therm_t1': 285.0,
                          'therm_e1': 0.2,
                          'therm_t2': 0.0,
                          'therm_e2': 0.0,
                          'therm_t3': 33.0,
                          'therm_e3': 0.01,
                          'vacair': 'vac',
                          'wmin': 300.0,
                          'wmax': 2500.0,
                          'wgrid_mode': 'fixed_wavelength_step',
                          'wdelta': self.step*1e3,
                          'wres': self.res,
                          'lsf_type': 'none',
                          'lsf_gauss_fwhm': 1.0,
                          'lsf_boxcar_fwhm': 0.,
                          'observatory': 'paranal'}
        
        #storage params
        self.update_sky = False
        self.emm = None
        self.trans = None
        self.wavelength = None
        self.fli = None
        self.airmass = None
        self.pwv = None
        
        #store initial FLI, airmass, pwv, and update internal parameters
        self.set_params(fli=fli, airmass=airmass, pwv=pwv)
                
    def _get_moon_sun_sep(self, fli=None):
        #convert FLI to moon sun separation
        return 180.-np.degrees(np.arccos(2*fli-1))
        
    def set_params(self, fli=None, airmass=None, pwv=None):
        #store values and update parameter dictionary
        if fli is not None:
            if fli != self.fli: #is this different than the current value:?
                sep = self._get_moon_sun_sep(fli)
                self.paramDict['moon_sun_sep'] = sep
                self.fli = fli
                self.update_sky = True
        if airmass is not None:
            if airmass != self.airmass: #is this different than the current value:?
                self.paramDict['airmass'] = airmass
                self.airmass = airmass
                self.update_sky = True
        if pwv is not None:
            if pwv != self.pwv: #is this different than the current value:?
                self.paramDict['pwv'] = pwv
                self.pwv = pwv
                self.update_sky = True
               
    def _calc_sky(self):
        #actually run the sky model
        parfile = 'sky_pars.json'
        skyfile = 'sky_spec.fits'

        #dump default parameters
        with open(parfile, 'w') as json_file:
            json.dump(self.paramDict, json_file)
        
        #run the skycalc call
        os.system('skycalc_cli -i {0} -o {1}/{2}'.format(parfile, os.getcwd(), skyfile))
        
        #pull in the sky spectrum
        with fits.open(skyfile) as sfile:
            tab = sfile[1].data
            self.wavelength = np.asarray(tab['LAM'], dtype=np.float)
            self.emm = np.asarray(tab['FLUX'], dtype=np.float)
            self.trans = np.asarray(tab['TRANS'], dtype=np.float)
        
        #update resolution in pixels
        self.res_pix = self.wavelength / self.res / self.step / 2.355
        
        os.remove(parfile)
        os.remove(skyfile)
        self.update_sky = False
            
    def __call__(self, fli=None, airmass=None, pwv=None):
        self.set_params(fli=fli, airmass=airmass, pwv=pwv)
        
        if self.update_sky:
            self._calc_sky()
        
        return self.wavelength, self.emm, self.trans
