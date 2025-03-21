import os
import sys

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)

import camb
import config

pars = camb.CAMBparams() #initialise the CAMBparams object, which contains all cosmological parameters and settings

Hubble = config.H0
baryons = config.ombh2
cdm = config.omch2
tilt = config.ns
zmax = config.zmax
kmax = config.kmax
extrap_kmax = config.extrap_kmax
clight = config.c

pars.set_cosmology(H0=Hubble, ombh2=baryons, omch2=cdm)   #define the cosmological model
pars.InitPower.set_params(ns=tilt)                      #set the primordial power spectrum parameters
background = camb.get_background(pars)                #compute the background cosmological evolution

#this gives us an interpolator which can be used to generate the Weyl power spectrum for any range of z and k
Weyl_power_spectra = camb.get_matter_power_interpolator(pars, zmax=zmax, kmax=kmax, zs=None,
hubble_units=False, k_hunit=False, var1=camb.model.Transfer_Weyl, var2=camb.model.Transfer_Weyl, extrap_kmax=extrap_kmax)

correlations_prefactor = -2*((clight*1e-3)**2) / (3 * (baryons + cdm) * 1e4)     