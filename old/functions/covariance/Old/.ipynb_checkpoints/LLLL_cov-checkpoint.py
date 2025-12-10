import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('xi2_LOS_plus_intp','xi1_LOS_plus_intp','xi32_LOS_plus_intp','xi2_LOS_minus_intp','xi1_LOS_minus_intp','xi32_LOS_minus_intp')

############################################################ 6.1 LLLL ########################################################################
##############################################################################################################################################

################################################# 6.1.1 LLLL noise covariance ################################################################

def generate_ncov_LLLL(distributions, sigma_noise, sigma_shape, b1, b2, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear autcorrelation function.

    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b (0 to 4) (irrelevant for this matrix, usually set to None)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution = distributions['LL']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    # Initialise the tables with their non-integral diagonal elements
    
    diagonal = (sigma_noise**2 * (sigma_noise**2 + 2 * xi1_LOS_plus_intp(0))
                * Omegatot / Omegas) / Nlens**2
    ncov_pp = np.diag(diagonal)
    ncov_mm = np.diag(diagonal)
    ncov_pm = np.zeros_like(ncov_pp)
    
    if not approx:
        
        # Define the integrands leading to the off-diagonal terms
        # we choose to include the differential elements, but not the normalisation

        def integrand_pp(coords):
            r1, r2, psi2 = coords
            r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))
            f = 2 * np.pi * r1 * r2 * xi2_LOS_plus_intp(r)
            return f

        def integrand_mm(coords):
            r1, r2, psi2 = coords
            r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))
            f = 2 * np.pi * r1 * r2 * xi2_LOS_plus_intp(r) * np.cos(4 * psi2)
            return f

        def integrand_pm(coords):
            r1, r2, psi2 = coords
            x = r1 - r2 * np.cos(psi2)
            y = -r2 * np.sin(psi2)
            r = np.sqrt(x**2 + y**2)
            psi = np.arctan2(y, x)
            f = 2 * np.pi * r1 * r2 * xi2_LOS_minus_intp(r) * np.cos(4 * psi)
            return f

        def integral_term(integrand, a1, a2, nsample):
            ranges = [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)]
            integral, err = monte_carlo_integrate(integrand, 
                ranges, num_samples = nsample)
            integral /= (Omegas[a1] * Omegas[a2]) # normalisation of differential elements
            err /= (Omegas[a1] * Omegas[a2]) 
            result = integral * 2 * sigma_noise**2 / Nlens
            err = err * 2 * sigma_noise**2 / Nlens
            return result, err
        
        # Load all necessary files once before the loop
        nsamples_pp = load_file("nsample_dicts/LLLL/ncovpp") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ncovpp") else None
        nsamples_mm = load_file("nsample_dicts/LLLL/ncovmm") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ncovmm") else None
        nsamples_pm = load_file("nsample_dicts/LLLL/ncovpm") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ncovpm") else None
        
        for a1 in range(Nbin):
            # pp and mm are symmetric, so loop from a1 to Nbin
            for a2 in range(a1, Nbin):
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_pp = int(nsamples_pp[str(a1) + str(a2)]) if nsamples_pp else nsamp
                nsample_mm = int(nsamples_mm[str(a1) + str(a2)]) if nsamples_mm else nsamp
        
                # Compute the integrals
                pp_integral, pp_err = integral_term(integrand_pp, a1, a2, nsample_pp)
                mm_integral, mm_err = integral_term(integrand_mm, a1, a2, nsample_mm)
                
                ncov_pp[a1, a2] += pp_integral
                ncov_pp[a2, a1] = ncov_pp[a1, a2]
                ncov_mm[a1, a2] += mm_integral
                ncov_mm[a2, a1] = ncov_mm[a1, a2]

                test_err(pp_err, ncov_pp[a1, a2], f'LLLL ncov pp angular bins {a1, a2}')
                test_err(mm_err, ncov_mm[a1, a2], f'LLLL ncov mm angular bins {a1, a2}')
        
            # pm is not symmetric, so loop over all a2
            for a2 in range(Nbin):
        
                # Use preloaded nsamples or fall back to nsamp
                nsample_pm = int(nsamples_pm[str(a1) + str(a2)]) if nsamples_pm else nsamp
        
                # Compute the integral
                pm_integral, pm_err = integral_term(integrand_pm, a1, a2, nsample_pm)
                
                ncov_pm[a1, a2] += pm_integral

                test_err(pm_err, ncov_pm[a1, a2], f'LLLL ncov pm angular bins {a1, a2}')
                
            
    # Make the full noise covariance matrix, with first xi_+ and then xi_-
    ncov_mp = np.transpose(ncov_pm)
    ncov = np.block([[ncov_pp, ncov_pm],
                     [ncov_mp, ncov_mm]])
    
    
    return ncov, ncov_pp, ncov_mm, ncov_pm, ncov_mp

################################################# 6.1.2 LLLL cosmic covariance ###############################################################

def generate_ccov_LLLL(distributions, b1, b2, approx=False):
    """
    Computes the ccontribution of cosmic variance in the covariance matrix
    of the LOS shear autcorrelation function.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b (0 to 4) (irrelevant for this matrix, usually set to None)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution = distributions['LL']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    # Initialise the blocks
    ccov_pp = np.zeros((Nbin, Nbin))
    ccov_mm = np.zeros((Nbin, Nbin))
    ccov_pm = np.zeros((Nbin, Nbin))
    
    
    # Define the integrands
    
    def integrand_pp(params):
        psi1, psi2, r1, r2, r3 = params
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        f = (xi2_LOS_plus_intp(r3) * xi2_LOS_plus_intp(r)
             + xi2_LOS_minus_intp(r3) * xi2_LOS_minus_intp(r) * np.cos(4 * psi)
            ) * 2 * np.pi * r3 * r1 * r2
        return f
    
    def integrand_mm(params):
        psi1, psi2, r1, r2, r3 = params
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        f = (xi2_LOS_plus_intp(r3) * xi2_LOS_plus_intp(r) * np.cos(4 * (psi1 - psi2))
             + xi2_LOS_minus_intp(r3) * xi2_LOS_minus_intp(r) * np.cos(4 * (psi - psi1 - psi2))
            ) * 2 * np.pi * r3 * r1 * r2
        return f
    
    def integrand_pm(params):
        psi1, psi2, r1, r2, r3 = params
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        f = (2 * xi2_LOS_minus_intp(r3) * xi2_LOS_plus_intp(r) * np.cos(4 * psi2)
            ) * 2 * np.pi * r3 * r1 * r2
        return f

    # Compute and add the integral contribution
    
    def integral_bins(integrand, a1, a2, nsampl):
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, r2_max)]
        integral, err = monte_carlo_integrate(integrand, ranges, nsampl)
        # normalisation of differential elements
        integral /= (Omegatot * Omegas[a1] * Omegas[a2])
        err /= (Omegatot * Omegas[a1] * Omegas[a2]) 
        return integral, err
        
    # Load all necessary files once before the loop
    nsamples_pp = load_file("nsample_dicts/LLLL/ccovpp") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ccovpp") else None
    nsamples_mm = load_file("nsample_dicts/LLLL/ccovmm") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ccovmm") else None
    nsamples_pm = load_file("nsample_dicts/LLLL/ccovpm") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ccovpm") else None
    
    for a1 in range(Nbin):
        for a2 in range(a1, Nbin): # pp and mm are symmetric
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pp = int(nsamples_pp[str(a1) + str(a2)]) if nsamples_pp else nsamp
            nsample_mm = int(nsamples_mm[str(a1) + str(a2)]) if nsamples_mm else nsamp
                     
            ccov_pp[a1, a2], pp_err = integral_bins(integrand_pp, a1, a2, nsample_pp)
            ccov_pp[a2, a1] = ccov_pp[a1, a2]
            ccov_mm[a1, a2], mm_err = integral_bins(integrand_mm, a1, a2, nsample_mm)
            ccov_mm[a2, a1] = ccov_mm[a1, a2]

            test_err(pp_err, ccov_pp[a1, a2], f'LLLL ccov pp angular bins {a1, a2}')

            test_err(mm_err, ccov_mm[a1, a2], f'LLLL ccov mm angular bins {a1, a2}')
            
        for a2 in range(Nbin): # while pm isn't in principle
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pm = int(nsamples_pm['00'][str(a1) + str(a2)]) if nsamples_pm else nsamp
            
            ccov_pm[a1, a2], pm_err = integral_bins(integrand_pm, a1, a2, nsample_pm)

            test_err(pm_err, ccov_pm[a1, a2], f'LLLL ccov pm angular bins {a1, a2}')
            
    
    # Make the full cosmic covariance matrix, with first xi_+ and then xi_-
    ccov_mp = np.transpose(ccov_pm)
    ccov = np.block([[ccov_pp, ccov_pm],
                     [ccov_mp, ccov_mm]])
    
    
    return ccov, ccov_pp, ccov_mm, ccov_pm, ccov_mp

################################################# 6.1.3 LLLL sparsity covariance #############################################################

def generate_scov_LLLL(distributions, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the LOS shear autcorrelation function.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b (0 to 4) (irrelevant for this matrix, usually set to None)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution = distributions['LL']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    # Preliminaries: integral of the correlation functions over angular bins
    
    xi2_plus_bins = []
    xi2_minus_bins = []
    
    xi2_plus_errs = []
    xi2_minus_errs = []
    
    for a in range(Nbin):
        
        # xi_plus
        
        def integrand(r):
            f = 2 * np.pi * r * xi2_LOS_plus_intp(r)
            return f
        
        integral, err = monte_carlo_integrate(integrand, [(rs[a], rs[a+1])], 2e1)
        integral /= Omegas[a]
        err /= Omegas[a]
        xi2_plus_bins.append(integral)
        xi2_plus_errs.append(err)
        
        # xi_minus
        
        def integrand(r):
            f = 2 * np.pi * r * xi2_LOS_minus_intp(r)
            return f
        
        integral, err = monte_carlo_integrate(integrand, [(rs[a], rs[a+1])], 2e1)
        integral /= Omegas[a]
        err /= Omegas[a]
        xi2_minus_bins.append(integral)
        xi2_minus_errs.append(err)
    
    
    # Plus - plus block
    
    scov_pp = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to the Krönecker delta
        # in the expression of SCov(+,+)

        f = (xi1_LOS_plus_intp(0)**2) * 2 * np.pi * r

        if not approx:
            f += (3 * xi1_LOS_plus_intp(r)**2
                 + xi1_LOS_minus_intp(r)**2
                ) * 2 * np.pi * r
            
        return f
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the other term in the expression of SCov(+,+)
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
            
        f = xi1_LOS_plus_intp(0) * xi2_LOS_plus_intp(r12) * 2 * np.pi * r1 * r2

        if not approx:
            
            f += 3 * xi32_LOS_plus_intp(r1) * xi32_LOS_plus_intp(r2) * 2 * np.pi * r1 * r2
            
        return f
      
    for a1 in range(Nbin):
        
        # term proportional to the Krönecker delta
        integral, err = monte_carlo_integrate(integrand_diag, [(rs[a1], rs[a1+1])], 1e2)
        integral /= Omegas[a1]
        err /= Omegas[a1]
        
        scov_pp[a1, a1] = (integral - 2 * xi2_plus_bins[a1]**2 
                          ) * Omegatot / Omegas[a1] / Nlens**2
        diag_err = (err*2 + (4 * xi2_plus_bins[a1] * xi2_plus_errs[a1])**2) * (Omegatot / Omegas[a1] / Nlens**2  )**2
        
        # term without the Krönecker delta
        for a2 in range(a1, Nbin):
            
            pp_err = 0

            if a1 == a2:
                pp_err += diag_err
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)], 1e4)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2])  
            
            scov_pp[a1, a2] += (integral
                                - 2 * xi2_plus_bins[a1] * xi2_plus_bins[a2]
                                ) * 2 / Nlens

            pp_err += (err**2
                                + (2 * xi2_plus_errs[a1] * xi2_plus_bins[a2])**2
                                + (2 * xi2_plus_bins[a1] * xi2_plus_errs[a2])**2
                                ) * (2 / Nlens)**2 

            pp_err = np.sqrt(pp_err)

            test_err(pp_err, scov_pp[a1, a2], f'LLLL scov pp angular bins {a1, a2}')
            
            scov_pp[a2, a1] = scov_pp[a1, a2]
    
    
    # Minus - minus block
    
    scov_mm = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to the Krönecker delta
        # in the expression of SCov(-,-)
        
        f = (xi1_LOS_plus_intp(0)**2) * 2 * np.pi * r

        if not approx:
            f += (3 * xi1_LOS_minus_intp(r)**2
             + xi1_LOS_plus_intp(r)**2
             ) * 2 * np.pi * r
            
        return f
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the other term in the expression of SCov(-,-)
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
    
        f = xi1_LOS_plus_intp(0) * xi2_LOS_plus_intp(r12) * np.cos(4 * psi2) * 2 * np.pi * r1 * r2

        if not approx:
            f += (3 * xi32_LOS_plus_intp(r1) * xi32_LOS_plus_intp(r2) 
                ) * 2 * np.pi * r1 * r2
        return f
    
    for a1 in range(Nbin):
        
        # term proportional to the Krönecker delta
        integral, err = monte_carlo_integrate(integrand_diag, [(rs[a1], rs[a1+1])], 1e2)
        integral /= Omegas[a1]
        err /= Omegas[a1]
        
        scov_mm[a1, a1] = (integral
                           - 2 * xi2_minus_bins[a1]**2 
                           ) * Omegatot / Omegas[a1] / Nlens**2
        diag_err = (err*2 + (4 * xi2_minus_bins[a1] * xi2_minus_errs[a1])**2) * (Omegatot / Omegas[a1] / Nlens**2  )**2
        
        # term without the Krönecker delta
        for a2 in range(a1, Nbin):
            
            mm_err = 0

            if a1 == a2:
                mm_err += diag_err
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)], 4e5)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2])  
            
            scov_mm[a1, a2] += (integral
                                - 2 * xi2_minus_bins[a1] * xi2_minus_bins[a2] 
                                ) * 2 / Nlens

            mm_err += (err**2
                                + (2 * xi2_minus_errs[a1] * xi2_minus_bins[a2])**2
                                + (2 * xi2_minus_bins[a1] * xi2_minus_errs[a2])**2
                                ) * (2 / Nlens)**2 

            mm_err = np.sqrt(mm_err)

            test_err(mm_err, scov_mm[a1, a2], f'LLLL scov mm angular bins {a1, a2}')
            
            scov_mm[a2, a1] = scov_mm[a1, a2]
    
    
    # Plus - minus block
    
    scov_pm = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to the Krönecker delta
        # in the expression of SCov(+,-)
        f = (4 * xi1_LOS_plus_intp(r) * xi1_LOS_minus_intp(r)
            ) * 2 * np.pi * r
        return f
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the other term in the expression of SCov(-,-)
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi12 = np.arctan2(y12, x12)
        
        f = xi1_LOS_plus_intp(0) * xi2_LOS_minus_intp(r12) * np.cos(4 * (psi12 - psi2)) * 2 * np.pi * r1 * r2

        if not approx:
        
            f += (3 * xi32_LOS_plus_intp(r1) * xi32_LOS_minus_intp(r2)
                ) * 2 * np.pi * r1 * r2
            
        return f
        
    # Load all necessary files once before the loop
    nsamples_pm = load_file("nsample_dicts/LLLL/ncovpm") if use_measured_samples and os.path.exists("nsample_dicts/LLLL/ncovpm") else None
    
    for a1 in range(Nbin):
        
        # term proportional to the Krönecker delta
        if not approx:
            integral, err = monte_carlo_integrate(integrand_diag, [(rs[a1], rs[a1+1])], 3e2)
        
        else:
            integral = 0
            err = 0
            
        integral /= Omegas[a1]
        err /= Omegas[a1]
        
        scov_pm[a1, a1] = (integral
                           - 2 * xi2_plus_bins[a1] * xi2_minus_bins[a1] 
                           ) * Omegatot / Omegas[a1] / Nlens**2

        diag_err = (err*2 + (2 * xi2_plus_errs[a1] * xi2_minus_bins[a1])**2 + (2 * xi2_plus_bins[a1] * xi2_minus_errs[a1])**2 ) * (Omegatot / Omegas[a1] / Nlens**2  )**2
        
        # term without the Krönecker delta
        for a2 in range(a1, Nbin):
            
            pm_err = 0

            if a1 == a2:
                pm_err += diag_err
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_pm = int(nsamples_pm[str(a1) + str(a2)]) if nsamples_pm else nsamp
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample_pm)
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            err /= (Omegas[a1] * Omegas[a2])
            
            scov_pm[a1, a2] += (integral
                                - 2 * xi2_plus_bins[a1] * xi2_minus_bins[a2] 
                                ) * 2 / Nlens

            pm_err += (err**2
                                + (2 * xi2_plus_errs[a1] * xi2_minus_bins[a2])**2
                                + (2 * xi2_plus_bins[a1] * xi2_minus_errs[a2])**2
                                ) * (2 / Nlens)**2 

            pm_err = np.sqrt(pm_err)

            test_err(pm_err, scov_pm[a1, a2], f'LLLL scov pm angular bins {a1, a2}')

            scov_pm[a2, a1] = scov_pm[a1, a2]
            
    
    # Make the full sparsity covariance matrix, with first xi_+ and then xi_-
    scov_mp = np.transpose(scov_pm)
    scov = np.block([[scov_pp, scov_pm],
                     [scov_mp, scov_mm]])
    
    
    return scov, scov_pp, scov_mm, scov_pm, scov_mp