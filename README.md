# Expose
This is meant to be a general purpose exposure time calculator for spectroscopy.  At the moment it doesn't really do much!

Current version: v0.1.0a0

# Installation
If you have git installed, then you can obtain the software like so:
```
git clone https:/github.com/jtmendel/expose.git expose
cd expose
python setup.py develop
```
This will compile the necessary routines and create a link to their location in your python directory (currently recommended since things are likely to change).  If you chose, you can also perform a standard installation, i.e:
```
python setup.py install
```
There are various options controlling *where* things are located, with perhaps the most germane being `--prefix=/path/to/installation` and `--install-scripts=/directory/in/your/path`.  Additional install options can be perused using:

```
python setup.py install --help
```
As well as Numpy and Scipy, you will need to have the following Python modules installed
* [Cython](https://cython.org/)
* [astropy](http://www.astropy.org/) - Python tools for astronomy.
* [skycalc_cli](https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html) - The command line interface to ESO's advance sky model.
* [python-fsps](http://dfm.io/python-fsps/current/) - Python interface for FSPS.  Will also need to install FSPS!

# Contents
* Description coming soon, I promise!
* `doc`: contains (eventually) a manual, installation instructions, and version history, blah blah blah.
