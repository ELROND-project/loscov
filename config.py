import numpy as np

use_approx = False                #approximate calculation
compute_correlations = True      #computes the correlation functions necessary for calculating the covariance matrices,
                                 #and saves them in a pickle file. Set to False if this has already been done, which
                                 #saves computation time

matrices_folder = 'matrices_5e10_v1'    #the name of the folder in which we store the resulting matrices

######################################## Euclid galaxy distribution ##########################################

binparams = {
           'redshifts': [0,0.4676,0.7194,0.9625,1.3319,3],
           'A': [3.0005,3.0003,3.0008,3.0014,3.0019],
           'alpha': 0.4710,
           'beta': 5.11843,
           'gamma': 0.7259
}

############################################# Survey stats ###################################################

sky_coverage = 15e3     #area of sky covered by the survey (deg^2)
Nlens = 1e5             #number of lenses (expect 1e5)
NGal = 2e9              #total number of galaxies (expect 2e9)
Nbina = 5

Omegatot = sky_coverage * (np.pi / 180)**2                                #sky coverage (in rad^2)
lens_density = Nlens / Omegatot                                           #the density of lenses in an angular bin (in rad^-2)
n_b = ( NGal / sky_coverage * (np.pi / 180)**2 ) / len(binparams['A'])    #number density of galaxies per redshift bin (in rad^-2)
r2_max = np.sqrt(Omegatot/np.pi)                                         #the maximum angle out to which we integrate (in rad)

Nbina_LL = 5               #number of angular separation bins for lenses
Nbina_Le = 5               #number of angular separation bins for galaxy shapes
Nbina_Lp = 5               #number of angular separation bins for galaxy positions

Thetamax_dist = 3e2     #maximum theta to calculate distributions inputted into covariance matrices (arcmin)

Thetamax_LL = Thetamax_dist             
Thetamax_Le = Thetamax_dist             
Thetamax_Lp = Thetamax_dist                

use_binscheme = False       #should we explicitly define the angular limits of each bin?

if use_binscheme:
    binscheme_LL = [1,2,3,4,5,6]        #these obviously need to change, but the idea will be to list the angle limits of the bins (rad)
    binscheme_Le = [1,2,3,4,5,6]
    binscheme_Lp = [1,2,3,4,5,6]

else:                                   #if we don't define these limits, the binning is calculated automatically according to the number of bins
    binscheme_LL = Nbina_LL 
    binscheme_Le = Nbina_Le
    binscheme_Lp = Nbina_Lp
    
########################################## cosmology #########################################################

H0=67.37       #Hubble constant
ombh2=0.0223   #baryon density parameter
omch2=0.1198   #dark matter density parameter
ns = 0.965     #primordial power spectrum
c = 3e8        #speed of light

zmax = 7     #the maximum redshift to which we compute the Weyl power spectrum
kmax = 5e2 #(inverse Mpc) the maximum wavenumber, ie. the smallest spatial scales to which we determine the power spectrum
extrap_kmax = 1e10 #the maximum k for extrapolation beyond kmax

############################################ noise ###########################################################

sigma_shape = np.sqrt(2) * 0.3                                            #the noise from the galaxy shapes on cosmic shear
sigma_noise = 0.05                                                       #noise on the LOS shear (expect 0.05)              

##################################### numerical stuff ########################################################

use_measured_samples = True

if Nbina_LL != 5 or Nbina_Le != 5 or Nbina_Lp != 5 or len(binparams['redshifts']) != 6:
    use_measured_samples = False
    print("Warning: unable to use measured number of samples because bin numbers are changed")


minsamp = int(1e7)      #minimum number of samples in the Monte Carlo integrator
maxsamp = int(5e10)     #maximum number of samples in the Monte Carlo integrator
nsamp = int(1e8)        #default number of samples in the Monte Carlo integrator
num_batches = 1000        #should be > maxsamp * 373 / (ram per node)
desired_error = 1       #percentage desired fractional error in integrals
confidence = 0.95       #confidence level for the error estimate (not currently used)
warning_level = 500     #level above which we print an integration error
total_error_threshold = 0.2  #the threshold for the total error on a term to be too high and to print a warning

Thetamin_arcmin = 1                              #minimum theta from which we calculate correlation functions (in arcmin)   
Thetamax = np.pi                                 #maximum theta to which we calculate correlation functions (in radians)
nTheta = 10000                                   #number of points used to compute the correlation function

lmax = 1e8
nl = 1000

############################################## shared variables ##############################################

# Shared variables
global_dict = {}

########################## adjust these if only interested in specific combinations ########################## 

# Define parameter ranges
b1_values = [0, 1, 2, 3, 4]
b2_values = [0, 1, 2, 3, 4]

cov_matrices_full = ['LpLe', 'LeLe', 'LpLp']   # Needs (b1, b2) 
cov_matrices_b1 = ['LLLp', 'LLLe']             # Needs only b1
cov_matrices_no_b = ['LLLL']                   # No (b1, b2) 

cov_types = ["ncov", "ccov", "scov"]