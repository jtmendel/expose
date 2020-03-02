#!/usr/bin/env python
import os, sys
import numpy
from os.path import join as pjoin
import shutil
import glob

try:
    from setuptools import setup, Extension, Command
    from setuptools.command.build_ext import build_ext as _build_ext
    from setuptools.command.build import build
except ImportError:
    from distutils.core import setup, Extension, Command
    from distutils.command.build_ext import build_ext as _build_ext
    from distutils.command.build import build

class CleanCommand(Command):
    """Custom distutils command to clean the .so and .pyc files."""

    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self._clean_me = []
        self._clean_trees = []
        self._clean_exclude = []

        for root, dirs, files in list(os.walk('pyspark')):
            for f in files:
                if f in self._clean_exclude:
                    continue
                if os.path.splitext(f)[-1] in ('.pyc', '.so', '.o',
                                               '.pyo',
                                               '.pyd', '.c', '.orig'):
                    self._clean_me.append(pjoin(root, f))
            for d in dirs:
                if d == '__pycache__':
                    self._clean_trees.append(pjoin(root, d))

        for d in ('build', 'dist', ):
            if os.path.exists(d):
                self._clean_trees.append(d)

    def finalize_options(self):
        pass

    def run(self):
        for clean_me in self._clean_me:
            try:
                os.unlink(clean_me)
            except Exception:
                pass
        for clean_tree in self._clean_trees:
            try:
                import shutil
                shutil.rmtree(clean_tree)
            except Exception:
                pass


try:
    from Cython.Build import cythonize
except ImportError:
    raise ImportError("Expose requires cython to install")


class build_ext(_build_ext):
    def build_extension(self, ext):
        _build_ext.build_extension(self, ext)
        

if __name__ == "__main__":

    include_dirs = ["include",
                    numpy.get_include(),
                   ]

    cmodules = []
    cmodules += [Extension("expose.utils.interpolation", 
                           ["expose/utils/interpolation.pyx"], 
                           include_dirs=include_dirs)]
    cmodules += [Extension("expose.utils.smoothing", 
                           ["expose/utils/smoothing.pyx"], 
                           include_dirs=include_dirs)]
    ext_modules = cythonize(cmodules)

    #scripts = ['scripts/'+file for file in os.listdir('scripts/')]  
    scripts = []  

    cmdclass = {'clean': CleanCommand,
                'build_ext': build_ext}

  
    with open('expose/_version.py') as f:
        exec(f.read())

    setup(
        name = "expose",
        url="https://github.com/jtmendel/expose",
        version= __version__,
        author="Trevor Mendel",
        author_email="trevor.mendel@anu.edu.au",
        ext_modules = ext_modules,
        cmdclass = cmdclass,
        scripts = scripts, 
        packages=["expose",
                  "expose.instruments",
                  "expose.telescopes",
                  "expose.sky",
                  "expose.sources",
                  "expose.utils"],
        license="LICENSE",
        description="Flexible, Universal exposure time calculator",
        package_data={"": ["README.md", "LICENSE"],
                      "expose": ["expose/data/mavis/PSF_5mas_370nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_380nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_390nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_400nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_450nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_500nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_550nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_700nm_EEProfile.dat",
                              "expose/data/mavis/PSF_5mas_950nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_380nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_390nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_400nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_450nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_500nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_550nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_700nm_EEProfile.dat",
                              "expose/data/mavis/PSF_10mas_950nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_370nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_380nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_390nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_400nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_450nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_500nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_550nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_700nm_EEProfile.dat",
                              "expose/data/mavis/PSF_20mas_950nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_370nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_380nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_390nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_400nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_450nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_500nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_550nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_700nm_EEProfile.dat",
                              "expose/data/mavis/PSF_30mas_950nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_370nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_380nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_390nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_400nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_450nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_500nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_550nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_700nm_EEProfile.dat",
                              "expose/data/mavis/PSF_40mas_950nm_EEProfile.dat",
                              "expose/data/mavis/mavis_AOM_throughput.csv",
                              "expose/data/muse/muse_throughput_WFM.txt",
                              "expose/data/muse/muse_throughput_NFM.txt",
                              "expose/data/ref_sky_dark.fits",
                              "expose/data/ref_sky_grey.fits",
                              "expose/data/ref_sky_bright.fits",
                              "expose/data/E2V_QE.csv",
			      "expose/data/UT4_M1_reflect.csv",
			      "expose/data/UT4_M2_reflect.csv",
			      "expose/data/UT4_M3_reflect.csv",
			      "expose/data/grating_eff.csv",
			      "expose/data/1280scicam_QE.csv",
                              "expose/data/kc_templates/elliptical_template.fits",
                              "expose/data/kc_templates/s0_template.fits",
                              "expose/data/kc_templates/sa_template.fits",
                              "expose/data/kc_templates/sb_template.fits",
                              "expose/data/kc_templates/sc_template.fits",
                              "expose/data/kc_templates/starb1_template.fits",
                              "expose/data/kc_templates/starb2_template.fits",
                              "expose/data/kc_templates/starb3_template.fits",
                              "expose/data/kc_templates/starb4_template.fits",
                              "expose/data/kc_templates/starb5_template.fits",
                              "expose/data/kc_templates/starb6_template.fits",
                              ]},
        include_package_data=True,
        zip_safe=False,
    )













