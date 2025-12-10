import numpy as np
import sys, platform, os

compute_correlations = False     #computes the correlation functions necessary for calculating the covariance matrices,
                                 #and saves them in a pickle file. Set to False if this has already been done, which
                                 #saves computation time
                                 #Note, this is automatically set to True if the correlations were calculated for different
                                 #redshift binning, but NB if the number of bins stays the same but the binscheme changes,
                                 #it should be manually set to True to ensure the old correlations are recalculated

##############################################################################################################
############################################# Survey stats ###################################################
##############################################################################################################

sky_coverage = 15e3     #area of sky covered by the survey (deg^2)
Nlens = 1e5             #number of lenses (expect 1e5)

NGal = 2e9              #total number of galaxies (expect 2e9)

######################################## Redshift distribution ##########################################

zmax_dist = 3           #the default maximum redshift being considered
Nbin_z = 6              #the default number of redshift bins

# redshift bin limits 
# binscheme_E = [0,0.4676,0.7194,0.9625,1.3319,3]
# binscheme_P = [0,0.4676,0.7194,0.9625,1.3319,3]

# Nbinz_E = len(binscheme_E)-1
# Nbinz_P = len(binscheme_P)-1

# zmax_E = binscheme_E[-1]
# zmax_P = binscheme_P[-1]

#automatically calculate the redshift bin limits
Nbinz_E = Nbin_z        #the number of redshift bins for galaxy shapes
Nbinz_P = Nbin_z      #the number of redshift bins for galaxy positions

binscheme_E = Nbinz_E
binscheme_P = Nbinz_P

zmax_E = zmax_dist
zmax_P = zmax_dist

######################################## Angular distribution ##########################################

supply_binscheme = False     #set it true if you want to manually set the binning, false and the binning will be optimised according to SNR
    
Thetamax_dist = 3e2     #maximum theta to calculate distributions inputted into covariance matrices (arcmin)

Thetamax_LL = Thetamax_dist             
Thetamax_LE = Thetamax_dist             
Thetamax_LP = Thetamax_dist 

Thetamax_LL_plus = Thetamax_LL
Thetamax_LL_minus = Thetamax_LL

Thetamax_LE_plus = Thetamax_LE
Thetamax_LE_minus = Thetamax_LE
    
Omegatot = sky_coverage * (np.pi / 180)**2                                #sky coverage (in rad^2)
lens_density = Nlens / Omegatot                                           #the density of lenses in an angular bin (in rad^-2)
n_b = ( NGal / sky_coverage * (np.pi / 180)**2 ) / Nbin_z                 #number density of galaxies per redshift bin (in rad^-2)
r2_max = np.sqrt(Omegatot/np.pi)                                          #the maximum theta used in integrals which would normally run from 0 to infty 

if supply_binscheme:

    Nbina = 6              #the default number of angular separation bins
    Nbina = int(Nbina)                                      #the maximum angle out to which we integrate (in rad)  
    
    # angular bin limits (radians)
    # binscheme_LL_plus = [0.0, 0.03902675, 0.05519216, 0.06759631, 0.0780535,  0.08726646]
    # binscheme_LL_minus = [0.0, 0.03902675, 0.05519216, 0.06759631, 0.0780535,  0.08726646]
    # binscheme_LE_plus = [0.0, 0.03902675, 0.05519216, 0.06759631, 0.0780535,  0.08726646]
    # binscheme_LE_minus = [0.0, 0.03902675, 0.05519216, 0.06759631, 0.0780535,  0.08726646]
    # binscheme_LP = [0.0, 0.03902675, 0.05519216, 0.06759631, 0.0780535,  0.08726646]
    
    # Nbina_LL_plus = len(binscheme_LL_plus) - 1           #number of angular separation bins for LL plus
    # Nbina_LL_minus = len(binscheme_LL_minus) - 1         #number of angular separation bins for LL minus
    # Nbina_LE_plus = len(binscheme_LE_plus) - 1           #number of angular separation bins for LE plus
    # Nbina_LE_minus = len(binscheme_LE_minus) - 1         #number of angular separation bins for LE minus
    # Nbina_LP = len(binscheme_LP) - 1                     #number of angular separation bins for LP
    
    # automatically calculate the binning according to the number of bins
    Nbina_LL_plus = Nbina                #number of angular separation bins for LL plus
    Nbina_LL_minus = Nbina               #number of angular separation bins for LL minus
    Nbina_LE_plus = Nbina                #number of angular separation bins for LE plus
    Nbina_LE_minus = Nbina               #number of angular separation bins for LE minus
    Nbina_LP = Nbina                     #number of angular separation bins for galaxy positions
    
    binscheme_LL_plus = Nbina_LL_plus 
    binscheme_LL_minus = Nbina_LL_minus 
    binscheme_LE_plus = Nbina_LE_plus 
    binscheme_LE_minus = Nbina_LE_minus
    binscheme_LP = Nbina_LP

else:

    SNR_goal = 8
    Nbin_max = 20          #this should dictate the maximum number of angular bins, but it won't be exact, and relies on an empirical relationship
    SNR_min = 2.5
    theta_resolution = 1000  #the number of thetas in the linspace with which we obtain the max theta and calculate the "total SNR"

    SNR_goal_LL_plus = 6
    Nbin_max_LL_plus =  Nbin_max
    SNR_min_LL_plus = SNR_min 

    SNR_goal_LL_minus = 6
    Nbin_max_LL_minus =  Nbin_max
    SNR_min_LL_minus = SNR_min

    SNR_goal_LE_plus = SNR_goal
    Nbin_max_LE_plus =  Nbin_max
    SNR_min_LE_plus = SNR_min   

    SNR_goal_LE_minus = SNR_goal
    Nbin_max_LE_minus =  Nbin_max
    SNR_min_LE_minus = SNR_min       

    SNR_goal_LP = SNR_goal
    Nbin_max_LP =  Nbin_max
    SNR_min_LP = SNR_min          

########################################## cosmology #########################################################

H0=67.37       #Hubble constant
h=H0/100
ombh2=0.0223   #baryon density parameter
omch2=0.1198   #dark matter density parameter
ns = 0.965     #primordial power spectrum
c = 3e8        #speed of light

Omega_M = (ombh2 + omch2)/(h**2)
Omega_L = 1 - Omega_M

zmax = 7     #the maximum redshift to which we compute the Weyl power spectrum
kmax = 5e2 #(inverse Mpc) the maximum wavenumber, ie. the smallest spatial scales to which we determine the power spectrum
extrap_kmax = 1e10 #the maximum k for extrapolation beyond kmax

############################################ noise ###########################################################

sigma_E = np.sqrt(2) * 0.3                                            #the noise from the galaxy shapes on cosmic shear
sigma_L = 0.05                                                        #noise on the LOS shear (expect 0.05)  

##################################### numerical stuff ########################################################

max_cpus = 512
nsamp = 1e7
Csamp = nsamp*10        #default number of samples in the Monte Carlo integrator for triple cosmic integrals
Nsamp = nsamp           #default number of samples in the Monte Carlo integrator for double noise/sparsity integrals
num_batches = 1000     #should be > maxsamp * 373 / (ram per node)
desired_error = 1       #percentage desired fractional error in integrals
confidence = 0.95       #confidence level for the error estimate (not currently used)
warning_level = 500     #level above which we print an integration error
total_error_threshold = 0.2  #the threshold for the total error on a term to be too high and to print a warning

Thetamin_arcmin = 0.1                            #minimum theta from which we calculate correlation functions (in arcmin)   
Thetamax = np.pi                                 #maximum theta to which we calculate correlation functions (in radians)
nTheta = 10000                                   #number of points used to compute the correlation function

lmax = 1e8
nl = 1000

################################################## Imports #####################################################

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from scipy import constants, special, integrate, stats
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from hankel import HankelTransform

camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower

import pickle
from multiprocessing import Pool, cpu_count
from multiprocessing import Process, Manager
from itertools import product
import inspect
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar, minimize_scalar

############################################### Cosmology #####################################################

# CAMB parameters
pars = camb.CAMBparams()                              #initialise the CAMBparams object, which contains all cosmological parameters and settings
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)   #define the cosmological model
pars.InitPower.set_params(ns=ns)                      #set the primordial power spectrum parameters
background = camb.get_background(pars)                #compute the background cosmological evolution

#this gives us an interpolator which can be used to generate the Weyl power spectrum for any range of z and k
Weyl_power_spectra = camb.get_matter_power_interpolator(pars, zmax=zmax, kmax=kmax, zs=None,
hubble_units=False, k_hunit=False, var1=model.Transfer_Weyl, var2=model.Transfer_Weyl, extrap_kmax=extrap_kmax)

############################################## shared variables ##############################################

# Shared variables
global_dict = {}

########################## adjust these if only interested in specific combinations ########################## 

cov_matrices_full = ['LELP', 'LELE', 'LPLP']   # Needs (b1, b2) 
cov_matrices_b1 = ['LLLP', 'LLLE']             # Needs only b1
cov_matrices_no_b = ['LLLL']                   # No (b1, b2) 

# cov_matrices_full = ['LELE']   # Needs (b1, b2) 
# cov_matrices_b1 = []             # Needs only b1
# cov_matrices_no_b = []                   # No (b1, b2) 

cov_types = ["ccov", "ncov"]    

####################### the suffix defining the folder names ###############################################

notes = 'nsamp=1e7' #anything particularly unique about a particular run (eg different redshift binning)
correlation_notes = ''    #needed only to specify that a particular binscheme has been used

def format_sci(n):
    return f'{n:.0e}'.replace('+00', '').replace('+0', '').replace('+', '').replace('-0', '-')

if supply_binscheme == True:

    suffix = f'Nlens={format_sci(Nlens)}_sigL={sigma_L}_Nbin_z={Nbin_z}_Nbina={Nbina}_{notes}'

if supply_binscheme == False:

    suffix = f'Nlens={format_sci(Nlens)}_sigL={sigma_L}_Nbin_z={Nbin_z}_SNR_goal={SNR_goal}_Nbin_max={Nbin_max}_{notes}'
    

if not os.path.exists(f'correlations_NE={Nbinz_E}_NP={Nbinz_P}{correlation_notes}'):
    compute_correlations = True      

