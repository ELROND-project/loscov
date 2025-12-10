import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from galaxy_distribution import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('xi2_LOS_plus_intp','xi1_LOS_plus_intp','xi32_LOS_plus_intp','xi2_LOS_minus_intp','xi1_LOS_minus_intp','xi32_LOS_minus_intp')
get_item('xi2_eps_plus_intp','xi1_eps_plus_intp','xi2_eps_minus_intp','xi1_eps_minus_intp')
get_item('xi2_LOS_eps_plus_intp', 'xi2_LOS_eps_minus_intp', 'xi32_LOS_eps2_plus_intp', 'xi32_LOS_eps2_minus_intp', 'xi32_LOS2_eps_plus_intp', 'xi32_LOS2_eps_minus_intp')

################################################## 6.3 LLLe #################################################
#############################################################################################################

####################################### 6.3.1 LLLe noise covariance #########################################

def generate_ncov_LLLe(distributions, sigma_noise, sigma_shape, b1, b2, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear - ellipticity correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Le']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina     #the number of angular bins for Le 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Le (in rad)
    
    # Initialise the blocks
    ncov_pp = np.zeros((Nbin1, Nbin2))
    ncov_mm = np.zeros((Nbin1, Nbin2))
    ncov_pm = np.zeros((Nbin1, Nbin2))
    ncov_mp = np.zeros((Nbin1, Nbin2))
    
    # Define the integrands leading to the off-diagonal terms
    # I'll include everything which isn't constant between the terms as the integrand

    def integrand_pp(params):
        r1, r2, psi2 = params
        r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))           #the equivalent of l-l'
           
        f = 2 * np.pi * r1 * r2 * sigma_noise**2 * xi2_LOS_eps_plus_intp[b1](r) / Nlens
            
        return f

    def integrand_pm(params):
        r1, r2, psi2 = params
        r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))           #the equivalent of l-l'
       
        f = 2 * np.pi * r1 * r2 * sigma_noise**2 * xi2_LOS_eps_minus_intp[b1](r) / Nlens
        
        return f

    def integrand_mp(params):
        r1, r2, psi2 = params
        r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))           #the equivalent of l-l'
       
        f = 2 * np.pi * r1 * r2 * sigma_noise**2 * xi2_LOS_eps_minus_intp[b1](r) * np.cos(4 * psi2) / Nlens
        
        return f

    def integrand_mm(params):
        r1, r2, psi2 = params
        r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))           #the equivalent of l-l'
       
        f = 2 * np.pi * r1 * r2 * sigma_noise**2 * xi2_LOS_eps_plus_intp[b1](r) * np.cos(4 * psi2) / Nlens
            
        return f
        
    def integral_term(integrand, a1, a2, nsampl):
        integral, err  = monte_carlo_integrate(integrand, [(rs1[a1], rs1[a1+1]), (rs2[a2], rs2[a2+1]),(0, 2*np.pi)], nsampl)
        integral /= (Omegas1[a1]**2) # normalisation of differential elements 
        err /= (Omegas1[a1]**2) 
        return integral, err
    
    # Compute and add the integral contribution

    # Load all necessary files once before the loop
    nsamples_pp = load_file("nsample_dicts/LLLe/ncovpp") if use_measured_samples and os.path.exists("nsample_dicts/LLLe/ncovpp") else None
    nsamples_mm = load_file("nsample_dicts/LLLe/ncovmm") if use_measured_samples and os.path.exists("nsample_dicts/LLLe/ncovmm") else None

    for a1 in range(Nbin1):
        for a2 in range(Nbin2):
    
            # Use preloaded nsamples or fall back to nsamp
            nsample_pp = int(nsamples_pp[str(b1) + str(0)][str(a1) + str(a2)]) if nsamples_pp else nsamp
            nsample_mm = int(nsamples_mm[str(b1) + str(0)][str(a1) + str(a2)]) if nsamples_mm else nsamp

            int_pp, pp_err = integral_term(integrand_pp, a1, a2, nsample_pp)
            int_mm, mm_err = integral_term(integrand_mm, a1, a2, nsample_mm)

            int_pm, pm_err = integral_term(integrand_pm, a1, a2, 8e2)
            int_mp, mp_err = integral_term(integrand_pm, a1, a2, 8e2)
            
            ncov_pp[a1, a2] += int_pp
            ncov_mm[a1, a2] += int_mm

            ncov_pm[a1, a2] += int_pm
            ncov_mp[a1, a2] += int_mp

            test_err(pp_err, ncov_pp[a1, a2], f'LLLe ncov pp redshift {b1} angular bins {a1, a2}')
            test_err(mm_err, ncov_mm[a1, a2], f'LLLe ncov mm redshifts {b1} angular bins {a1, a2}')
            test_err(pm_err, ncov_pm[a1, a2], f'LLLe ncov pm redshifts {b1} angular bins {a1, a2}')
            test_err(mp_err, ncov_mp[a1, a2], f'LLLe ncov mp redshifts {b1} angular bins {a1, a2}')
            
    
    # Make the full noise covariance matrix, with first xi_+ and then xi_-
    ncov = np.block([[ncov_pp, ncov_pm],
                     [ncov_mp, ncov_mm]])
    
    
    return ncov, ncov_pp, ncov_mm, ncov_pm, ncov_mp
    
######################################## 6.3.2 LLLe cosmic covariance ###############################################

def generate_ccov_LLLe(distributions, b1, b2, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - ellipticity correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Le']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina     #the number of angular bins for Le 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Le (in rad)
    
    
    # Initialise the blocks
    ccov_pp = np.zeros((Nbin1, Nbin2))
    ccov_pm = np.zeros((Nbin1, Nbin2))
    ccov_mp = np.zeros((Nbin1, Nbin2))
    ccov_mm = np.zeros((Nbin1, Nbin2))
    
    
    # Define the integrands 
    
    def integrand_pp(params):
        psi1, psi2, r1, r2, r3 = params
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        
        x32 = r3 + r2 * np.cos(psi2)
        y32 = r2 * np.sin(psi2)
        r32 = (x32**2 + y32**2)**0.5
        psi32 = np.arctan2(y32, x32)
        
        x31 = r3 - r1 * np.cos(psi1)
        y31 = - r1 * np.sin(psi1)
        r31 = (x31**2 + y31**2)**0.5
        psi31 = np.arctan2(y31, x31)
        
        f = (2* xi2_LOS_plus_intp(r1) * xi2_LOS_eps_plus_intp[b1](r2)
             + xi2_LOS_plus_intp(r3) * xi2_LOS_eps_plus_intp[b1](r)
             + xi2_LOS_minus_intp(r3) * xi2_LOS_eps_minus_intp[b1](r) * np.cos(4 * psi)
             + xi2_LOS_plus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32)
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_minus_intp[b1](r32)*np.cos(4*(psi31+psi32)) #might want to check the + vs - in the cos, but I think it's all good
            ) * np.pi * r3 * r1 * r2 #the factor of 2 from the 2pi cancels with the factor of 1/2 in the expression for CCov
        return f
    
    def integrand_pm(params):
        psi1, psi2, r1, r2, r3 = params
        
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        
        x32 = r3 + r2 * np.cos(psi2)
        y32 = r2 * np.sin(psi2)
        r32 = (x32**2 + y32**2)**0.5
        psi32 = np.arctan2(y32, x32)
        
        x31 = r3 - r1 * np.cos(psi1)
        y31 = - r1 * np.sin(psi1)
        r31 = (x31**2 + y31**2)**0.5
        psi31 = np.arctan2(y31, x31)
        
        f = (2* xi2_LOS_plus_intp(r1) * xi2_LOS_eps_plus_intp[b1](r2)
             + xi2_LOS_plus_intp(r3) * xi2_LOS_eps_minus_intp[b1](r) * np.cos(4 * (psi2-psi))
             + xi2_LOS_minus_intp(r3) * xi2_LOS_eps_plus_intp[b1](r) * np.cos(4 * psi2)
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32) * np.cos(4 * (psi2+psi31))
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32)*np.cos(4*(psi2-psi32)) #might want to check the + vs - in the cos, but I think it's all good
            ) * np.pi * r3 * r1 * r2 #the factor of 2 from the 2pi cancels with the factor of 1/2 in the expression for CCov
        return f
    
    def integrand_mp(params):
        psi1, psi2, r1, r2, r3 = params
        
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        
        x32 = r3 + r2 * np.cos(psi2)
        y32 = r2 * np.sin(psi2)
        r32 = (x32**2 + y32**2)**0.5
        psi32 = np.arctan2(y32, x32)
        
        x31 = r3 - r1 * np.cos(psi1)
        y31 = - r1 * np.sin(psi1)
        r31 = (x31**2 + y31**2)**0.5
        psi31 = np.arctan2(y31, x31)
        
        f = (2 * xi2_LOS_minus_intp(r1) * xi2_LOS_eps_plus_intp[b1](r2)               #weird factor of 2 stuff going on
             + xi2_LOS_plus_intp(r3) * xi2_LOS_eps_minus_intp[b1](r) * np.cos(4 * (psi1-psi))
             + xi2_LOS_minus_intp(r3) * xi2_LOS_eps_plus_intp[b1](r) * np.cos(4 * psi2)
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32) * np.cos(4 * (psi2+psi31))
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32)*np.cos(4*(psi1+psi31)) 
            ) * np.pi * r3 * r1 * r2 #the factor of 2 from the 2pi cancels with the factor of 1/2 in the expression for CCov
        return f
    
    def integrand_mm(params):
        psi1, psi2, r1, r2, r3 = params
        
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        
        x32 = r3 + r2 * np.cos(psi2)
        y32 = r2 * np.sin(psi2)
        r32 = (x32**2 + y32**2)**0.5
        psi32 = np.arctan2(y32, x32)
        
        x31 = r3 - r1 * np.cos(psi1)
        y31 = - r1 * np.sin(psi1)
        r31 = (x31**2 + y31**2)**0.5
        psi31 = np.arctan2(y31, x31)
        
        f = (2 * xi2_LOS_minus_intp(r1) * xi2_LOS_eps_minus_intp[b1](r2)
             + xi2_LOS_plus_intp(r3) * xi2_LOS_eps_plus_intp[b1](r) * np.cos(4 * (psi1-psi2))
             + xi2_LOS_minus_intp(r3) * xi2_LOS_eps_minus_intp[b1](r) * np.cos(4 * (psi-psi1-psi2))
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32) * np.cos(4 * (psi1-psi2))
             + xi2_LOS_minus_intp(r31) * xi2_LOS_eps_plus_intp[b1](r32)*np.cos(4*(psi32-psi31-psi1-psi2)) 
            ) * np.pi * r3 * r1 * r2 #the factor of 2 from the 2pi cancels with the factor of 1/2 in the expression for CCov
        return f
    

    # Compute and add the integral contribution
    
    def integral_bins(integrand, a1, a2, nnsamp):
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs1[a1], rs1[a1+1]), (rs2[a2], rs2[a2+1]), (0, r2_max)]
        integral, err = monte_carlo_integrate(integrand, ranges, nnsamp)
        # normalisation of differential elements
        integral /= (Omegatot * Omegas1[a1] * Omegas2[a2]) 
        err /= (Omegatot * Omegas1[a1] * Omegas2[a2])  
        return integral, err
    
    for a1 in range(Nbin1):
        for a2 in range(a1, Nbin2): # pp and mm are symmetric
                     
            ccov_pp[a1, a2], pp_err = integral_bins(integrand_pp, a1, a2, 4e3)
            ccov_pp[a2, a1] = ccov_pp[a1, a2]
            ccov_mm[a1, a2], mm_err = integral_bins(integrand_mm, a1, a2, 2e2)
            ccov_mm[a2, a1] = ccov_mm[a1, a2]

            test_err(pp_err, ccov_pp[a1, a2], f'LLLe ccov pp redshift {b1} angular bins {a1, a2}')
            test_err(mm_err, ccov_mm[a1, a2], f'LLLe ccov mm redshifts {b1} angular bins {a1, a2}')
            
        for a2 in range(Nbin2): # while pm and mp aren't in principle
        
            ccov_pm[a1, a2], pm_err = integral_bins(integrand_pm, a1, a2, 2e3)
            ccov_mp[a1, a2], mp_err = integral_bins(integrand_mp, a1, a2, 2e2)
            
            test_err(pm_err, ccov_pm[a1, a2], f'LLLe ccov pm redshifts {b1} angular bins {a1, a2}')
            test_err(mp_err, ccov_mp[a1, a2], f'LLLe ccov mp redshifts {b1} angular bins {a1, a2}')
            
    
    # Make the full cosmic covariance matrix, with first xi_+ and then xi_-

    ccov = np.block([[ccov_pp, ccov_pm],
                     [ccov_mp, ccov_mm]])
    
    
    return ccov, ccov_pp, ccov_mm, ccov_pm, ccov_mp

######################################## 6.3.3 LLLe sparsity covariance ###############################################

def generate_scov_LLLe(distributions, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the LOS shear - ellipticity correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Le']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina     #the number of angular bins for Le 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Le (in rad)

    
####################################### Plus - plus block ###########################################
    
    scov_pp = np.zeros((Nbin1, Nbin2))

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(+,+)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        
        f = xi1_LOS_plus_intp(0) * xi2_LOS_eps_plus_intp[b1](r12) * 2 * np.pi * r1 * r2

        if not approx:

            f += (3 * xi32_LOS_plus_intp(r1) * xi32_LOS_eps2_plus_intp[b1](r2)
             - 4 * xi2_LOS_plus_intp(r1) * xi2_LOS_eps_plus_intp[b1](r2) 
            ) * 2 * np.pi * r1 * r2
        
        return f

    for a1 in range(Nbin1):

        for a2 in range(Nbin2):
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs2[a2], rs2[a2+1]), (rs1[a1], rs1[a1+1]), (0, 2*np.pi)], 7e3)
            integral /= (Omegas1[a1] * Omegas2[a2]) # normalise
            err /= (Omegas1[a1] * Omegas2[a2]) 
            
            scov_pp[a1, a2] += (integral) / (2 * Nlens)
            err /= (2 * Nlens)
            
            test_err(err, scov_pp[a1, a2], f'LLLe scov pp redshift {b1} angular bins {a1, a2}')
    
    
####################################### Minus - minus block ###########################################
    
    scov_mm = np.zeros((Nbin1, Nbin2))
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(-,-)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)

        f = xi1_LOS_plus_intp(0) * xi2_LOS_eps_minus_intp[b1](r12) * np.cos(4 * (psi-psi2)) * 2 * np.pi * r1 * r2

        if not approx:
            f += (3 * xi32_LOS_plus_intp(r1) * xi32_LOS_eps2_minus_intp[b1](r2) 
             - 4 * xi2_LOS_minus_intp(r1) * xi2_LOS_eps_minus_intp[b1](r2) 
            ) * 2 * np.pi * r1 * r2
        
        return f
        
    nsamples_mm = load_file("nsample_dicts/LLLe/scovmm") if use_measured_samples and os.path.exists("nsample_dicts/LLLe/scovmm") else None

    for a1 in range(Nbin1):

        for a2 in range(Nbin2):
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_mm = int(nsamples_mm[str(b1)][str(a1) + str(a2)]) if nsamples_mm else nsamp
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs1[a1], rs1[a1+1]),(rs2[a2], rs2[a2+1]), (0, 2*np.pi)], nsample_mm)
            integral /= (Omegas1[a1] * Omegas2[a2]) # normalise
            err /= (Omegas1[a1] * Omegas2[a2]) 
            
            scov_mm[a1, a2] += (integral) / (2 * Nlens)
            err /= (2 * Nlens)
            
            test_err(err, scov_mm[a1, a2], f'LLLe scov mm redshift {b1} angular bins {a1, a2}')
    
    
####################################### Plus - minus block ###########################################
    
    scov_pm = np.zeros((Nbin1, Nbin2))

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(+,-)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)

        f = xi1_LOS_plus_intp(0) * xi2_LOS_eps_minus_intp[b1](r12) * np.cos(4 * (psi-psi2)) * 2 * np.pi * r1 * r2

        if not approx:
            f = (3 * xi32_LOS_plus_intp(r1) * xi32_LOS_eps2_minus_intp[b1](r2)
             - 4 * xi2_LOS_plus_intp(r1) * xi2_LOS_eps_minus_intp[b1](r2)
            ) * 2 * np.pi * r1 * r2
        
        return f
        
    nsamples_pm = load_file("nsample_dicts/LLLe/scovpm") if use_measured_samples and os.path.exists("nsample_dicts/LLLe/scovpm") else None

    for a1 in range(Nbin1):

        for a2 in range(Nbin2):
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pm = int(nsamples_pm[str(b1)][str(a1) + str(a2)]) if nsamples_pm else nsamp
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs1[a1], rs1[a1+1]), (rs2[a2], rs2[a2+1]), (0, 2*np.pi)], nsample_pm)
            integral /= (Omegas1[a1] * Omegas2[a2]) # normalise
            err /= (Omegas1[a1] * Omegas2[a2]) 
            
            scov_pm[a1, a2] += (integral) / (2 * Nlens)
            err /= (2 * Nlens)
            
            test_err(err, scov_pm[a1, a2], f'LLLe scov pp redshift {b1} angular bins {a1, a2}')
    
    
####################################### Minus - plus block ###########################################
    
    scov_mp = np.zeros((Nbin1, Nbin2))

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(-,+)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)

        f = xi1_LOS_plus_intp(0) * xi2_LOS_eps_minus_intp[b1](r12) * np.cos(4 * (psi-psi2)) * 2 * np.pi * r1 * r2 
        
        if not approx:
        
            f += (3 * xi32_LOS_plus_intp(r1) * xi32_LOS_eps2_minus_intp[b1](r2)
                 - 4 * xi2_LOS_minus_intp(r1) * xi2_LOS_eps_plus_intp[b1](r2) 
                ) * 2 * np.pi * r1 * r2
        return f
        
    nsamples_mp = load_file("nsample_dicts/LLLe/scovmp") if use_measured_samples and os.path.exists("nsample_dicts/LLLe/scovmp") else None

    for a1 in range(Nbin1):

        for a2 in range(Nbin2):
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_mp = int(nsamples_mp[str(b1)][str(a1) + str(a2)]) if nsamples_mp else nsamp
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs1[a1], rs1[a1+1]), (rs2[a2], rs2[a2+1]), (0, 2*np.pi)], nsample_mp)
            integral /= (Omegas1[a1] * Omegas2[a2]) # normalise
            err /= (Omegas1[a1] * Omegas2[a2]) 
            
            scov_mp[a1, a2] += (integral) / (2 * Nlens)
            err /= (2 * Nlens)
            
            test_err(err, scov_mp[a1, a2], f'LLLe scov pp redshift {b1} angular bins {a1, a2}')
    
    # Make the full sparsity covariance matrix, with first xi_+ and then xi_-
    scov = np.block([[scov_pp, scov_pm],
                     [scov_mp, scov_mm]])
    
    
    return scov, scov_pp, scov_mm, scov_pm, scov_mp