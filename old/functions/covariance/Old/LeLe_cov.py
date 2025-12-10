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
get_item('xi2_LOS_eps_plus_intp', 'xi2_LOS_eps_minus_intp', 'xi32_LOS_eps2_plus_intp', 'xi32_LOS_eps2_minus_intp', 'xi32_LOS2_eps_plus_intp', 'xi32_LOS2_eps_minus_intp', 'xi1_LOS_eps_plus_intp', 'xi1_LOS_eps_minus_intp')

############################################################## 6.2 LeLe ######################################################################
##############################################################################################################################################

################################################### 6.2.1 LeLe noise covariance ##############################################################

def generate_ncov_LeLe(distributions, sigma_noise, sigma_shape, b1, b2, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear - ellipticity correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """

    distribution = distributions['Le']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    Ngal1    = get_ngal(b1)        #G_b in the math - the number of galaxies in redshift bin b
    Ngal2    = get_ngal(b2)        #G_{b'} in the math - the number of galaxies in redshift bin b'
    
    # Initialise the tables with their non-integral diagonal elements

    if b1 == b2:
    
        diagonal = ( Omegatot / ( 2 * Nlens * Ngal1 * Omegas) ) * (sigma_noise**2 * sigma_shape**2 
                       +xi1_LOS_plus_intp(0)*sigma_shape**2 + xi1_eps_plus_intp[b1](0)*sigma_noise**2)

    else:
        
        diagonal = np.zeros(Nbin)
    
    ncov_pp = np.diag(diagonal)
    ncov_mm = np.diag(diagonal)

    #note that, like in the autocorrelations case, the pm and mp covariance matrices have no diagonal terms
    ncov_pm = np.zeros_like(ncov_pp)
    ncov_mp = np.zeros_like(ncov_pp)
    
    if not approx:
        
        # Define the integrands
        # I'll include everything which isn't constant between the terms as the integrand

        def integrand_pp(params):
            r1, r2, psi2 = params  # Unpack the input array
            r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))           #the equivalent of l-l'

            if b1 == b2:
                f = 2 * np.pi * r1 * r2 * (sigma_shape**2 * xi2_LOS_plus_intp(r)  + sigma_noise**2 * Ngal2 * xi2_eps_plus_intp[b1][b2](r) / Nlens)
            else:
                f = 2 * np.pi * r1 * r2 * ( sigma_noise**2 * Ngal2 * xi2_eps_plus_intp[b1][b2](r) / Nlens)
            
            return f

        def integrand_pm(params):
            r1, r2, psi2 = params  # Unpack the input array
            x = r1 - r2 * np.cos(psi2)
            y = - r2 * np.sin(psi2)
            r = np.sqrt(x**2 + y**2)
            psi = np.arctan2(y, x) 

            if b1 == b2:
                f = 2 * np.pi * r1 * r2 * np.cos(4 * (psi-psi2)) * (sigma_shape**2 * xi2_LOS_minus_intp(r) + sigma_noise**2 * Ngal2 * xi2_eps_minus_intp[b1][b2](r) / Nlens)
            else:
                f = 2 * np.pi * r1 * r2 * np.cos(4 * (psi-psi2)) * (sigma_noise**2 * Ngal2 * xi2_eps_minus_intp[b1][b2](r) / Nlens)
            
            return f

        def integrand_mp(params):
            r1, r2, psi2 = params  # Unpack the input array
            x = r1 - r2 * np.cos(psi2)
            y = - r2 * np.sin(psi2)
            r = np.sqrt(x**2 + y**2)
            psi = np.arctan2(y, x)

            if b1 == b2:
                f = 2 * np.pi * r1 * r2 * np.cos(4 * psi) * (sigma_shape**2 * xi2_LOS_minus_intp(r) + sigma_noise**2 * Ngal2 * xi2_eps_minus_intp[b1][b2](r) / Nlens)
            else:
                f = 2 * np.pi * r1 * r2 * np.cos(4 * psi) * (sigma_noise**2 * Ngal2 * xi2_eps_minus_intp[b1][b2](r) / Nlens)
            
            return f

        def integrand_mm(params):
            r1, r2, psi2 = params  # Unpack the input array
            r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))

            if b1 == b2:
                f = 2 * np.pi * r1 * r2 * np.cos(4 * psi2) * (sigma_shape**2 * xi2_LOS_plus_intp(r)  + sigma_noise**2 * Ngal2 * xi2_eps_plus_intp[b1][b2](r) / Nlens)
            else:
                f = 2 * np.pi * r1 * r2 * np.cos(4 * psi2) * (sigma_noise**2 * Ngal2 * xi2_eps_plus_intp[b1][b2](r) / Nlens)
                
            return f
            
        def integral_term(integrand, a1, a2, nsample):
            integral, err = monte_carlo_integrate(integrand, [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample)
            integral /= (Omegas[a1]*Omegas[a2]) 
            err /= (Omegas[a1]*Omegas[a2])  
            result = integral / (2*Ngal1) # shared factors in the front
            res_err = err / (2*Ngal1)
            return result, res_err
        
        # Compute and add the integral contribution

        # Load all necessary files once before the loop
        nsamples_mm = load_file("nsample_dicts/LeLe/ncovmm") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovmm") else None
        nsamples_pm = load_file("nsample_dicts/LeLe/ncovpm") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovpm") else None
        nsamples_mp = load_file("nsample_dicts/LeLe/ncovmp") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovmp") else None
        
        for a1 in range(Nbin):
            for a2 in range(a1, Nbin): # pp and mm are symmetric
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_mm = int(nsamples_mm[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mm else nsamp

                pp_integral, pp_err = integral_term(integrand_pp, a1, a2, 8e3)
                mm_integral, mm_err = integral_term(integrand_mm, a1, a2, nsample_mm)
                
                ncov_pp[a1, a2] += pp_integral
                ncov_pp[a2, a1] =  ncov_pp[a1, a2]
                ncov_mm[a1, a2] += mm_integral
                ncov_mm[a2, a1] =  ncov_mm[a1, a2]

                test_err(pp_err, ncov_pp[a1, a2], f'LeLe ncov pp redshifts {b1, b2} angular bins {a1, a2}')
                test_err(mm_err, ncov_mm[a1, a2], f'LeLe ncov mm redshifts {b1, b2} angular bins {a1, a2}')

            for a2 in range(Nbin): # while pm isn't
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_pm = int(nsamples_pm[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_pm else nsamp
                nsample_mp = int(nsamples_mp[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mp else nsamp

                pm_integral, pm_err = integral_term(integrand_pm, a1, a2, nsample_pm)
                mp_integral, mp_err = integral_term(integrand_mp, a1, a2, nsample_mp)

                ncov_pm[a1, a2] += pm_integral
                ncov_mp[a1, a2] += mp_integral

                test_err(pm_err, ncov_pm[a1, a2], f'LeLe ncov pm redshifts {b1, b2} angular bins {a1, a2}')
                test_err(mp_err, ncov_mp[a1, a2], f'LeLe ncov mp redshifts {b1, b2} angular bins {a1, a2}')
            
    
    # Make the full noise covariance matrix, with first xi_+ and then xi_-
    ncov = np.block([[ncov_pp, ncov_pm],
                     [ncov_mp, ncov_mm]])
    
    return ncov, ncov_pp, ncov_mm, ncov_pm, ncov_mp

################################################## 6.2.2 LeLe cosmic covariance ##############################################################

def generate_ccov_LeLe(distributions, b1, b2, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - ellipticity correlation functions.

    lens_dist      : statistics relating to the lens distribution
    galaxy_dist    : statistics relating to the galaxy distribution
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    """
    
    distribution = distributions['Le']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    Ngal1    = get_ngal(b1)        #G_b in the math - the number of galaxies in redshift bin b
    Ngal2    = get_ngal(b2)        #G_{b'} in the math - the number of galaxies in redshift bin b'
    
    # Initialise the blocks
    ccov_pp = np.zeros((Nbin, Nbin))
    ccov_pm = np.zeros((Nbin, Nbin))
    ccov_mp = np.zeros((Nbin, Nbin))
    ccov_mm = np.zeros((Nbin, Nbin))
    
    
    # Define the integrands (complete from here)
    
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
        
        f = (xi2_LOS_plus_intp(r3) * xi2_eps_plus_intp[b1][b2](r)
             + xi2_LOS_minus_intp(r3) * xi2_eps_minus_intp[b1][b2](r) * np.cos(4 * psi)
             + xi2_LOS_eps_plus_intp[b2](r31) * xi2_LOS_eps_plus_intp[b1](r32)
             + xi2_LOS_eps_minus_intp[b2](r31) * xi2_LOS_eps_minus_intp[b1](r32)*np.cos(4*(psi31+psi32)) 
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
        
        f = (xi2_LOS_plus_intp(r3) * xi2_eps_minus_intp[b1][b2](r) * np.cos(4 * (psi2 - psi) )
             + xi2_LOS_minus_intp(r3) * xi2_eps_plus_intp[b1][b2](r) * np.cos(4 * psi2)
             + xi2_LOS_eps_minus_intp[b2](r31) * xi2_LOS_eps_plus_intp[b1](r32) * np.cos(4 * (psi2+psi31)) 
             + xi2_LOS_eps_plus_intp[b2](r31) * xi2_LOS_eps_minus_intp[b1](r32) * np.cos(4 * (psi2-psi32)) 
            ) * np.pi * r3 * r1 * r2 #factor of 2 cancels
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
        
        f = (xi2_LOS_plus_intp(r3) * xi2_eps_minus_intp[b1][b2](r) * np.cos(4 * (psi1 - psi) )
             + xi2_LOS_minus_intp(r3) * xi2_eps_plus_intp[b1][b2](r) * np.cos(4 * psi1)
             + xi2_LOS_eps_minus_intp[b2](r31) * xi2_LOS_eps_plus_intp[b1](r32) * np.cos(4 * (psi1+psi31)) #again, maybe weird that it's plus
             + xi2_LOS_eps_plus_intp[b2](r31) * xi2_LOS_eps_minus_intp[b1](r32) * np.cos(4 * (psi1-psi32)) 
            ) * np.pi * r3 * r1 * r2 #factor of 2 cancels
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
        
        f = (xi2_LOS_plus_intp(r3) * xi2_eps_plus_intp[b1][b2](r) * np.cos(4 * (psi1 - psi2) )
             + xi2_LOS_minus_intp(r3) * xi2_eps_minus_intp[b2][b1](r) * np.cos(4 * (psi-psi1-psi2))
             + xi2_LOS_eps_plus_intp[b2](r31) * xi2_LOS_eps_plus_intp[b1](r32) * np.cos(4 * (psi1 - psi2) )
             + xi2_LOS_eps_minus_intp[b2](r31) * xi2_LOS_eps_minus_intp[b1](r32) * np.cos(4 * (psi32-psi31-psi1-psi2) ) #the asymmetry in the psi32-psi31 worries me
            ) * np.pi * r3 * r1 * r2 #factor of 2 cancels
        return f
    

    # Compute and add the integral contribution

    # Load all necessary files once before the loop
    nsamples_pp = load_file("nsample_dicts/LeLe/ncovpp") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovpp") else None
    nsamples_mm = load_file("nsample_dicts/LeLe/ncovmm") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovmm") else None
    nsamples_pm = load_file("nsample_dicts/LeLe/ncovpm") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovpm") else None
    nsamples_mp = load_file("nsample_dicts/LeLe/ncovmp") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/ncovmp") else None
    
    def integral_bins(integrand, a1, a2, nsampl):
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, r2_max)]
        integral, err = monte_carlo_integrate(integrand, ranges, nsampl)
        # normalisation of differential elements
        integral /= (Omegatot * Omegas[a1] * Omegas[a2]) 
        err /= (Omegatot * Omegas[a1] * Omegas[a2]) 
        return integral, err
    
    for a1 in range(Nbin):
        for a2 in range(a1, Nbin): # pp and mm are symmetric
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pp = int(nsamples_pp[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_pp else nsamp
            nsample_mm = int(nsamples_mm[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mm else nsamp
                     
            ccov_pp[a1, a2], pp_err = integral_bins(integrand_pp, a1, a2, nsample_pp)
            ccov_pp[a2, a1] = ccov_pp[a1, a2]
            ccov_mm[a1, a2], mm_err = integral_bins(integrand_mm, a1, a2, nsample_mm)
            ccov_mm[a2, a1] = ccov_mm[a1, a2]

            test_err(pp_err, ccov_pp[a1, a2], f'LeLe ccov pp redshifts {b1, b2} angular bins {a1, a2}')
            test_err(mm_err, ccov_mm[a1, a2], f'LeLe ccov mm redshifts {b1, b2} angular bins {a1, a2}')
            
        for a2 in range(Nbin): # while pm and mp aren't in principle
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pm = int(nsamples_pm[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_pm else nsamp
            nsample_mp = int(nsamples_mp[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mp else nsamp
            
            ccov_pm[a1, a2], pm_err = integral_bins(integrand_pm, a1, a2, nsample_pm)
            ccov_mp[a1, a2], mp_err = integral_bins(integrand_mp, a1, a2, nsample_mp)

            test_err(pm_err, ccov_pm[a1, a2], f'LeLe ccov pm redshifts {b1, b2} angular bins {a1, a2}')
            test_err(mp_err, ccov_mp[a1, a2], f'LeLe ccov mp redshifts {b1, b2} angular bins {a1, a2}')
            
    
    # Make the full cosmic covariance matrix, with first xi_+ and then xi_-

    ccov = np.block([[ccov_pp, ccov_pm],
                     [ccov_mp, ccov_mm]])
    
    
    return ccov, ccov_pp, ccov_mm, ccov_pm, ccov_mp

################################################## 6.2.3 LeLe sparsity covariance #############################################################

def generate_scov_LeLe(distributions, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the LOS shear - ellipticity correlation functions.

    lens_dist      : statistics relating to the lens distribution
    galaxy_dist    : statistics relating to the galaxy distribution
    b1             : the galaxy redshift bin b (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to include the integral terms in the covariance matrices
    """
    
    distribution = distributions['Le']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    Ngal1    = get_ngal(b1)        #G_b in the math - the number of galaxies in redshift bin b
    Ngal2    = get_ngal(b2)        #G_{b'} in the math - the number of galaxies in redshift bin b'

######################## plus or minus squared integrals appearing throughout #######################
        
    # xi_plus
        
    def xi2_plus_integrand(r):
        f = 2 * np.pi * r * xi2_LOS_eps_plus_intp[b1](r)
        return f
        
    # xi_minus
        
    def xi2_minus_integrand(r):
        f = 2 * np.pi * r * xi2_LOS_eps_minus_intp[b1](r)
        return f
    
    
####################################### Plus - plus block ###########################################
    
    scov_pp = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to both the Krönecker deltas
        # in the expression of SCov(+,+)
        f = (3 * xi1_LOS_eps_plus_intp[b1](r)**2
             + xi1_LOS_eps_minus_intp[b1](r)**2
            ) * 2 * np.pi * r
        return f

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(+,+)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        
        f = xi1_LOS_plus_intp(0) * xi2_eps_plus_intp[b1][b2](r12) * 2 * np.pi * r1 * r2

        if not approx:
            f += (3 * xi32_LOS_eps2_plus_intp[b1](r1) * xi32_LOS_eps2_plus_intp[b2](r2)
             - 2 * xi2_LOS_eps_plus_intp[b1](r1) * xi2_LOS_eps_plus_intp[b2](r2) ) * 2 * np.pi * r1 * r2
        
        return f

    def integrandbb(params):
        r1, r2, psi2 = params
        # integrand of the term proportional to the Krönecker delta for redshift binning, but not the other one
        # in the expression of SCov(+,+)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        
        f = (3 * xi32_LOS2_eps_plus_intp[b1](r1) * xi32_LOS2_eps_plus_intp[b2](r2)
             - 2 * xi2_LOS_eps_plus_intp[b1](r1) * xi2_LOS_eps_plus_intp[b2](r2)
             + xi1_eps_plus_intp[b1](0) * xi2_LOS_plus_intp(r12) 
            ) * 2 * np.pi * r1 * r2
        return f
        
    nsamples_pp = load_file("nsample_dicts/LeLe/scovpp") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovpp") else None
    nsamples_ppbb = load_file("nsample_dicts/LeLe/scovppbb") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovppbb") else None
      
    for a1 in range(Nbin):

        if b1 == b2:
            # term proportional to the Krönecker delta for a and b
            ranges = [(rs[a1], rs[a1+1])]
            integral = xi1_LOS_plus_intp(0) * xi1_eps_plus_intp[b1](0)
            
            if not approx:
                integrall, err1 = monte_carlo_integrate(integrand_diag, ranges, 1e3)
                integral2square, err2 = monte_carlo_integrate(xi2_plus_integrand, ranges, 2e1)
                
                integral += integrall - 2*integral2square**2

                diag_err = err1**2 + (4*integral2square*err2)**2
            
            integral /= Omegas[a1]
            diag_err /= (Omegas[a1])**2
            
            scov_pp[a1, a1] = integral * Omegatot / ( 2 * Omegas[a1] * Nlens * Ngal1 )
            
            diag_err = diag_err * (Omegatot / ( 2 * Omegas[a1] * Nlens * Ngal1 ))**2

        # term without the Krönecker delta for a
        for a2 in range(a1, Nbin):
            pp_err2 = 0
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pp = int(nsamples_pp[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_pp else nsamp
            
            ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
            integral, err = monte_carlo_integrate(integrand, ranges, nsample_pp)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2]) 
            
            scov_pp[a1, a2] += (integral) / (2 * Nlens)
            pp_err2 += (err / (2*Nlens))**2

            #term with the Krönecker delta for b
            if b1 == b2 and not approx:                                         #the division by G_b makes this whole term very small
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_ppbb = int(nsamples_ppbb[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_ppbb else nsamp
                
                ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
                integral, err = monte_carlo_integrate(integrandbb, ranges, nsample_ppbb)
                integral /= (Omegas[a1] * Omegas[a2]) # normalise
                err /= (Omegas[a1] * Omegas[a2])
            
                scov_pp[a1, a2] += (integral) / (2 * Ngal1)
                pp_err2 += (err / (2 * Ngal1))**2

            if a1 == a2 and b1 == b2:
                pp_err2 += diag_err

            pp_err = np.sqrt(pp_err2)
            
            test_err(pp_err, scov_pp[a1, a2], f'LeLe scov pp redshifts {b1, b2} angular bins {a1, a2}')
            
            scov_pp[a2, a1] = scov_pp[a1, a2]
    
    
####################################### Minus - minus block ###########################################
    
    scov_mm = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to both the Krönecker deltas
        # in the expression of SCov(-,-)
        f = (3 * xi1_LOS_eps_minus_intp[b1](r)**2
             + xi1_LOS_eps_plus_intp[b1](r)**2
            ) * 2 * np.pi * r
        return f

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(-,-)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5

        f = xi1_LOS_plus_intp(0) * xi2_eps_plus_intp[b1][b2](r12) * np.cos(4 * psi2) * 2 * np.pi * r1 * r2

        if not approx:
            f += (3 * xi32_LOS_eps2_minus_intp[b1](r1) * xi32_LOS_eps2_minus_intp[b2](r2) 
             - 2 * xi2_LOS_eps_minus_intp[b1](r1) * xi2_LOS_eps_minus_intp[b2](r2)
            ) * 2 * np.pi * r1 * r2
        return f

    def integrandbb(params):
        r1, r2, psi2 = params
        # integrand of the term proportional to the Krönecker delta for redshift binning, but not the other one
        # in the expression of SCov(-,-)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        
        f = (3 * xi32_LOS2_eps_minus_intp[b1](r1) * xi32_LOS2_eps_minus_intp[b2](r2)
             - 2 * xi2_LOS_eps_minus_intp[b1](r1) * xi2_LOS_eps_minus_intp[b2](r2)
             + xi1_eps_plus_intp[b1](0) * xi2_LOS_plus_intp(r12) * np.cos(4 * psi2)
            ) * 2 * np.pi * r1 * r2
        return f
        
    nsamples_mm = load_file("nsample_dicts/LeLe/scovmm") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovmm") else None
    nsamples_mmbb = load_file("nsample_dicts/LeLe/scovmmbb") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovmmbb") else None
      
    for a1 in range(Nbin):

        if b1 == b2:
            # term proportional to the Krönecker delta for a and b
            ranges = [(rs[a1], rs[a1+1])]
            integral = xi1_LOS_plus_intp(0) * xi1_eps_plus_intp[b1](0)
            
            if not approx:
                integrall, err1 = monte_carlo_integrate(integrand_diag, ranges, 3e2)
                integral2square, err2 = monte_carlo_integrate(xi2_minus_integrand, ranges, 3e1)
                
                integral += integrall - 2*integral2square**2 

                diag_err += err1**2 + (4*integral2square*err2)**2
            
            integral /= Omegas[a1]
            diag_err /= (Omegas[a1])**2
        
            scov_mm[a1, a1] = integral * Omegatot / ( 2 * Omegas[a1] * Nlens * Ngal1 )
            diag_err = diag_err * (Omegatot / ( 2 * Omegas[a1] * Nlens * Ngal1 ))**2

        # term without the Krönecker delta for a
        for a2 in range(a1, Nbin):
            mm_err2 = 0
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_mm = int(nsamples_mm[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mm else nsamp
            
            ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
            integral, err = monte_carlo_integrate(integrand, ranges, nsample_mm)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2]) # normalise
            
            scov_mm[a1, a2] += (integral) / (2 * Nlens)
            mm_err2 += (err / (2 * Nlens))**2

            #term with the Krönecker delta for b
            if b1 == b2 and not approx:                                         #the division by G_b makes this whole term very small
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_mmbb = int(nsamples_mmbb[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mmbb else nsamp
                
                ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
                integral, err = monte_carlo_integrate(integrandbb, ranges, nsample_mmbb)
                integral /= (Omegas[a1] * Omegas[a2]) # normalise
            
                scov_mm[a1, a2] += (integral) / (2 * Ngal1)
                mm_err2 += (err / (2 * Ngal1))**2

            if a1 == a2 and b1 == b2:
                mm_err2 += diag_err

            mm_err = np.sqrt(mm_err2)
            
            test_err(mm_err, scov_pp[a1, a2], f'LeLe scov mm redshifts {b1, b2} angular bins {a1, a2}')

            scov_mm[a2, a1] = scov_mm[a1, a2]
    
    
####################################### Plus - minus block ###########################################
    
    scov_pm = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to both the Krönecker deltas
        # in the expression of SCov(+,-)
        f = (xi1_LOS_eps_plus_intp[b1](r) * xi1_LOS_eps_minus_intp[b1](r) 
            ) * 2 * np.pi * r
        return f

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(+,-)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)

        f = xi1_LOS_plus_intp(0) * xi2_eps_minus_intp[b1][b2](r12) * np.cos(4 * (psi-psi2)) * 2 * np.pi * r1 * r2

        if not approx:
            f += (3 * xi32_LOS_eps2_plus_intp[b1](r1) * xi32_LOS_eps2_minus_intp[b2](r2)
                 - 2 * xi2_LOS_eps_plus_intp[b1](r1) * xi2_LOS_eps_minus_intp[b2](r2) 
                ) * 2 * np.pi * r1 * r2
        return f

    def integrandbb(params):
        r1, r2, psi2 = params
        # integrand of the term proportional to the Krönecker delta for redshift binning, but not the other one
        # in the expression of SCov(+,-)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)
        
        f = (3 * xi32_LOS2_eps_plus_intp[b1](r1) * xi32_LOS2_eps_minus_intp[b2](r2) 
             - 2 * xi2_LOS_eps_plus_intp[b1](r1) * xi2_LOS_eps_minus_intp[b2](r2) 
             + xi1_eps_plus_intp[b1](0) * xi2_LOS_minus_intp(r12) * np.cos(4 * (psi-psi2))   
            ) * 2 * np.pi * r1 * r2
        return f
        
    nsamples_pm = load_file("nsample_dicts/LeLe/scovpm") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovpm") else None
      
    for a1 in range(Nbin):  

        if b1 == b2 and not approx:

            integral, err = monte_carlo_integrate(integrand_diag, [(rs[a1], rs[a1+1])], 3e2)
            integral2, err2 = monte_carlo_integrate(xi2_plus_integrand, [(rs[a1], rs[a1+1])], 2e1)
            integral3, err3 = monte_carlo_integrate(xi2_minus_integrand, [(rs[a1], rs[a1+1])], 3e1)
            integral -= 2*integral2*integral3
            err = err**2
            err += (2*err2*err3)**2
            integral /= Omegas[a1]
            err /= (Omegas[a1])**2
        
            scov_pm[a1, a1] = 2 * integral * Omegatot / ( Omegas[a1] * Nlens * Ngal1 )
            diag_err = err * (2 * Omegatot / ( Omegas[a1] * Nlens * Ngal1 ))**2
        
        # term without the Krönecker delta for a
        for a2 in range(a1, Nbin):
           
            pm_err2 = 0
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pm = int(nsamples_pm[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_pm else nsamp
            
            ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
            integral, err = monte_carlo_integrate(integrand, ranges, nsample_pm)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2])
            
            scov_pm[a1, a2] += (integral) / (2 * Nlens)
            pm_err2 += (err / (2 * Nlens))**2

            if b1 == b2 and not approx:
                #term with the Krönecker delta for b
                
                ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
                integral, err = monte_carlo_integrate(integrandbb, ranges, 5e3)
                integral /= (Omegas[a1] * Omegas[a2]) # normalise
                err /= (Omegas[a1] * Omegas[a2]) 
            
                scov_pm[a1, a2] += (integral) / (2 * Ngal1)
                pm_err2 += (err / (2 * Ngal1))**2

            if a1 == a2 and b1 == b2:
                pm_err2 += diag_err

            pm_err = np.sqrt(pm_err2)
            
            test_err(pm_err, scov_pm[a1, a2], f'LeLe scov pm redshifts {b1, b2} angular bins {a1, a2}')

            scov_pm[a2, a1] = scov_pm[a1, a2]
    
    
####################################### Minus - plus block ###########################################
    
    scov_mp = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to both the Krönecker deltas
        # in the expression of SCov(-,+)
        f = (xi1_LOS_eps_plus_intp[b1](r) * xi1_LOS_eps_minus_intp[b1](r)
            ) * 2 * np.pi * r
        return f

    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the term which doesn't feature any Krönecker deltas
        # in the expression of SCov(-,+)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)

        f = xi1_LOS_plus_intp(0) * xi2_eps_minus_intp[b1][b2](r12) * np.cos(4 * (psi)) * 2 * np.pi * r1 * r2

        if not approx:
            f += (3 * xi32_LOS_eps2_minus_intp[b1](r1) * xi32_LOS_eps2_plus_intp[b2](r2)
                 - 2 * xi2_LOS_eps_plus_intp[b2](r2) * xi2_LOS_eps_minus_intp[b1](r1) 
                ) * 2 * np.pi * r1 * r2
        return f

    def integrandbb(params):
        r1, r2, psi2 = params
        # integrand of the term proportional to the Krönecker delta for redshift binning, but not the other one
        # in the expression of SCov(-,+)
        
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi = np.arctan2(y12,x12)
        
        f = (3 * xi32_LOS2_eps_minus_intp[b1](r1) * xi32_LOS2_eps_plus_intp[b2](r2) 
             + xi1_eps_plus_intp[b1](0) * xi2_LOS_minus_intp(r12) * np.cos(4 * (psi))   
            ) * 2 * np.pi * r1 * r2
        return f
        
    nsamples_mp = load_file("nsample_dicts/LeLe/scovmp") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovmp") else None
    nsamples_mpbb = load_file("nsample_dicts/LeLe/scovmpbb") if use_measured_samples and os.path.exists("nsample_dicts/LeLe/scovmpbb") else None
      
    for a1 in range(Nbin):

        if b1 == b2 and not approx:
            # term proportional to the Krönecker delta for a and b
            ranges = [(rs[a1], rs[a1+1])]
            integral, err = monte_carlo_integrate(integrand_diag, ranges, 3e2)
            integral2, err1 = monte_carlo_integrate(xi2_plus_integrand, ranges, 2e1)
            integral3, err2 = monte_carlo_integrate(xi2_minus_integrand, ranges, 3e1)
            integral -= 2*integral2*integral3
            integral /= Omegas[a1]
            err = err**2
            err += (2*err2*err3)**2
            err /= (Omegas[a1])**2
        
            scov_mp[a1, a1] = 2 * integral * Omegatot / ( Omegas[a1] * Nlens * Ngal1 )
            diag_err = err * (2 * Omegatot / ( Omegas[a1] * Nlens * Ngal1 ))**2

        
        # term without the Krönecker delta for a
        for a2 in range(a1, Nbin):

            mp_err2 = 0
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_mp = int(nsamples_mp[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mp else nsamp
            
            ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
            integral, err = monte_carlo_integrate(integrand, ranges, nsample_mp)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2])
            
            scov_mp[a1, a2] += (integral) / (2 * Nlens)
            mp_err2 += (err / (2 * Nlens))**2

            #term with the Krönecker delta for b
            if b1 == b2 and not approx:
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_mpbb = int(nsamples_mpbb[str(b1) + str(b2)][str(a1) + str(a2)]) if nsamples_mpbb else nsamp
                
                ranges = [(0, 2*np.pi), (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])]
                integral, err = monte_carlo_integrate(integrandbb, ranges, 5e3)
                integral /= (Omegas[a1] * Omegas[a2]) # normalise
                err /= (Omegas[a1] * Omegas[a2])
            
                scov_mp[a1, a2] += (integral) / (2 * Ngal1)
                mp_err2 += (err / (2 * Ngal1))**2

            if a1 == a2 and b1 == b2:
                mp_err2 += diag_err

            mp_err = np.sqrt(mp_err2)
            
            test_err(mp_err, scov_mp[a1, a2], f'LeLe scov mp redshifts {b1, b2} angular bins {a1, a2}')

            scov_mp[a2, a1] = scov_pm[a1, a2]
            
    
    # Make the full sparsity covariance matrix, with first xi_+ and then xi_-
    scov = np.block([[scov_pp, scov_pm],
                     [scov_mp, scov_mm]])
    
    
    return scov, scov_pp, scov_mm, scov_pm, scov_mp