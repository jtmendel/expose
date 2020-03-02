from __future__ import print_function
import numpy as np

__all__ = ['VLT', 'DREAMS', 'telescope']

class telescope:
    """
    Should contain general routines for a given telescope.
    Note that the intent is for this to be passed to a *specfic*
    telescope class later on."""
    def __init__(self):
        pass
    
    def _psf(self):
        pass

    
class VLT(telescope):
    def __init__(self):
        #initialize the model base
        telescope.__init__(self)

        self.area = 48.32507025
        
        #include some mirror stuff here so area is actually calculated
        #include PSF calculation as well?  could do

class DREAMS(telescope):
    def __init__(self):
        #initialize the model base
        telescope.__init__(self)

        self.area = np.pi*(0.5**2 - 0.226**2)
        
        #include some mirror stuff here so area is actually calculated
        #include PSF calculation as well?  could do

