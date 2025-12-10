import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

############################################################ 6.1 eeee ########################################################################
##############################################################################################################################################

################################################# 6.1.1 eeee noise covariance ################################################################

def generate_ncov_eeee(lens_dist, sigma_noise, sigma_shape, b1, b2, b3, b4, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear autcorrelation function.

    lens_dist      : statistics relating to the lens distribution
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b (0 to 4) 
    b2             : the galaxy redshift bin b' (0 to 4) 
    b3             : the galaxy redshift bin b'' (0 to 4) 
    b4             : the galaxy redshift bin b''' (0 to 4) 
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    Nbin     = lens_dist.Nbin
    Nlens    = lens_dist.Nlens
    Omegas   = lens_dist.Omegas
    Omegatot = lens_dist.Omegatot
    rs       = lens_dist.limits
    
    # Initialise the tables with their non-integral diagonal elements
    
    diagonal = (sigma_noise**2 * (sigma_noise**2 + 2 * xi1_eps_plus_intp[b1](0))
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
            f = 2 * np.pi * r1 * r2 * xi2_eps_plus_intp(r)
            return f

        def integrand_mm(coords):
            r1, r2, psi2 = coords
            r = np.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * np.cos(psi2))
            f = 2 * np.pi * r1 * r2 * xi2_eps_plus_intp(r) * np.cos(4 * psi2)
            return f

        def integrand_pm(coords):
            r1, r2, psi2 = coords
            x = r1 - r2 * np.cos(psi2)
            y = -r2 * np.sin(psi2)
            r = np.sqrt(x**2 + y**2)
            psi = np.arctan2(y, x)
            f = 2 * np.pi * r1 * r2 * xi2_eps_minus_intp(r) * np.cos(4 * psi)
            return f

        def integral_term(integrand, a1, a2):
            ranges = [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)]
            integral, err = monte_carlo_integrate(integrand, 
                ranges)
            integral /= (Omegas[a1] * Omegas[a2]) # normalisation of differential elements
            result = integral * 2 * sigma_noise**2 / Nlens
            return result
        
        # Compute and add the integral contribution
    
        for a1 in range(Nbin):
            for a2 in range(a1, Nbin): # pp and mm are symmetric

                ncov_pp[a1, a2] += integral_term(integrand_pp, a1, a2)
                ncov_pp[a2, a1] =  ncov_pp[a1, a2]
                ncov_mm[a1, a2] += integral_term(integrand_mm, a1, a2)
                ncov_mm[a2, a1] =  ncov_mm[a1, a2]

            for a2 in range(Nbin): # while pm isn't

                ncov_pm[a1, a2] += integral_term(integrand_pm, a1, a2)
            
    
    # Make the full noise covariance matrix, with first xi_+ and then xi_-
    ncov_mp = np.transpose(ncov_pm)
    ncov = np.block([[ncov_pp, ncov_pm],
                     [ncov_mp, ncov_mm]])
    
    
    return ncov, ncov_pp, ncov_mm, ncov_pm

################################################# 6.1.2 LLLL cosmic covariance ###############################################################

def generate_ccov_LLLL(lens_dist, b1, b2, approx=False):
    """
    Computes the ccontribution of cosmic variance in the covariance matrix
    of the eps shear autcorrelation function.

    lens_dist      : statistics relating to the lens distribution
    b1             : the galaxy redshift bin b (0 to 4) (irrelevant for this matrix, usually set to None)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    Nbin     = lens_dist.Nbin
    Nlens    = lens_dist.Nlens
    Omegas   = lens_dist.Omegas
    Omegatot = lens_dist.Omegatot
    rs       = lens_dist.limits
    
    
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
        f = (xi2_eps_plus_intp(r3) * xi2_eps_plus_intp(r)
             + xi2_eps_minus_intp(r3) * xi2_eps_minus_intp(r) * np.cos(4 * psi)
            ) * 2 * np.pi * r3 * r1 * r2
        return f
    
    def integrand_mm(params):
        psi1, psi2, r1, r2, r3 = params
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        psi = np.arctan2(y, x)
        f = (xi2_eps_plus_intp(r3) * xi2_eps_plus_intp(r) * np.cos(4 * (psi1 - psi2))
             + xi2_eps_minus_intp(r3) * xi2_eps_minus_intp(r) * np.cos(4 * (psi - psi1 - psi2))
            ) * 2 * np.pi * r3 * r1 * r2
        return f
    
    def integrand_pm(params):
        psi1, psi2, r1, r2, r3 = params
        x = r3 + r2 * np.cos(psi2) - r1 * np.cos(psi1)
        y = r2 * np.sin(psi2) - r1 * np.sin(psi1)
        r = (x**2 + y**2)**0.5
        f = (2 * xi2_eps_plus_intp(r3) * xi2_eps_plus_intp(r) * np.cos(4 * psi2)
            ) * 2 * np.pi * r3 * r1 * r2
        return f

    # Compute and add the integral contribution
    
    def integral_bins(integrand, a1, a2):
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, (Omegatot/np.pi)**0.5)]
        integral, err = monte_carlo_integrate(integrand, ranges)
        # normalisation of differential elements
        integral /= (Omegatot * Omegas[a1] * Omegas[a2]) 
        return integral
    
    for a1 in range(Nbin):
        print("Computing row", a1)
        for a2 in range(a1, Nbin): # pp and mm are symmetric
            print("Computing column", a2)
                     
            ccov_pp[a1, a2] = integral_bins(integrand_pp, a1, a2)
            print(ccov_pp[a1, a2])
            ccov_pp[a2, a1] = ccov_pp[a1, a2]
            ccov_mm[a1, a2] = integral_bins(integrand_mm, a1, a2)
            ccov_mm[a2, a1] = ccov_mm[a1, a2]
            
        for a2 in range(lens_dist.Nbin): # while pm isn't in principle
            
            ccov_pm[a1, a2] = integral_bins(integrand_pm, a1, a2)
            
    
    # Make the full cosmic covariance matrix, with first xi_+ and then xi_-
    ccov_mp = np.transpose(ccov_pm)
    ccov = np.block([[ccov_pp, ccov_pm],
                     [ccov_mp, ccov_mm]])
    
    
    return ccov, ccov_pp, ccov_mm, ccov_pm

################################################# 6.1.3 LLLL sparsity covariance #############################################################

def generate_scov_LLLL(lens_dist, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the eps shear autcorrelation function.

    lens_dist      : statistics relating to the lens distribution
    b1             : the galaxy redshift bin b (0 to 4) (irrelevant for this matrix, usually set to None)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    Nbin     = lens_dist.Nbin
    Nlens    = lens_dist.Nlens
    Omegas   = lens_dist.Omegas
    Omegatot = lens_dist.Omegatot
    rs       = lens_dist.limits
    
    if approx:
        kill = 0
    else:
        kill = 1
    
    # Preliminaries: integral of the correlation functions over angular bins
    
    xi2_plus_bins = []
    xi2_minus_bins = []
    
    for a in range(Nbin):
        
        # xi_plus
        
        def integrand(r):
            f = 2 * np.pi * r * xi2_eps_plus_intp(r)
            return f
        
        integral, err = monte_carlo_integrate(integrand, [(rs[a], rs[a+1])])
        integral /= Omegas[a]
        xi2_plus_bins.append(integral)
        
        # xi_minus
        
        def integrand(r):
            f = 2 * np.pi * r * xi2_eps_minus_intp(r)
            return f
        
        integral, err = monte_carlo_integrate(integrand, [(rs[a], rs[a+1])])
        integral /= Omegas[a]
        xi2_minus_bins.append(integral)
    
    
    # Plus - plus block
    
    scov_pp = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to the Krönecker delta
        # in the expression of SCov(+,+)
        f = (xi1_eps_plus_intp(0)**2
             + 3 * xi1_eps_plus_intp(r)**2 * kill
             + xi1_eps_minus_intp(r)**2 * kill
            ) * 2 * np.pi * r
        return f
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the other term in the expression of SCov(+,+)
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        f = (3 * xi32_eps_plus_intp(r1) * xi32_eps_plus_intp(r2) * kill
             + xi1_eps_plus_intp(0) * xi2_eps_plus_intp(r12)
            ) * 2 * np.pi * r1 * r2
        return f
      
    for a1 in range(Nbin):
        
        # term proportional to the Krönecker delta
        integral, err = monte_carlo_integrate(integrand_diag, [(rs[a], rs[a+1])])
        integral /= Omegas[a1]
        
        scov_pp[a1, a1] = (integral - 2 * xi2_plus_bins[a1]**2 #* kill # maybe not
                          ) * Omegatot / Omegas[a1] / Nlens**2
        
        # term without the Krönecker delta
        for a2 in range(a1, Nbin):
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)])
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            
            scov_pp[a1, a2] += (integral
                                - 2 * xi2_plus_bins[a1] * xi2_plus_bins[a2] #* kill # maybe not
                                ) * 2 / Nlens
            scov_pp[a2, a1] = scov_pp[a1, a2]
    
    
    # Minus - minus block
    
    scov_mm = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to the Krönecker delta
        # in the expression of SCov(-,-)
        f = (xi1_eps_plus_intp(0)**2
             + 3 * xi1_eps_minus_intp(r)**2 * kill
             + xi1_eps_plus_intp(r)**2 * kill
             ) * 2 * np.pi * r
        return f
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the other term in the expression of SCov(-,-)
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        f = (3 * xi32_eps_plus_intp(r1) * xi32_eps_plus_intp(r2) * kill
             + xi1_eps_plus_intp(0) * xi2_eps_plus_intp(r12) * np.cos(4 * psi2)
            ) * 2 * np.pi * r1 * r2
        return f
    
    for a1 in range(Nbin):
        
        # term proportional to the Krönecker delta
        integral, err = monte_carlo_integrate(integrand_diag, [(rs[a1], rs[a1+1])])
        integral /= Omegas[a1]
        
        scov_mm[a1, a1] = (integral
                           - 2 * xi2_minus_bins[a1]**2 * kill # maybe not
                           ) * Omegatot / Omegas[a1] / Nlens**2
        
        # term without the Krönecker delta
        for a2 in range(a1, Nbin):
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)])
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            
            scov_mm[a1, a2] += (integral
                                - 2 * xi2_minus_bins[a1] * xi2_minus_bins[a2] * kill # maybe not
                                ) * 2 / Nlens
            scov_mm[a2, a1] = scov_mm[a1, a2]
    
    
    # Plus - minus block
    
    scov_pm = np.zeros((Nbin, Nbin))
    
    def integrand_diag(r):
        # integrand of the term proportional to the Krönecker delta
        # in the expression of SCov(+,-)
        f = (4 * xi1_eps_plus_intp(r) * xi1_eps_minus_intp(r) * kill
            ) * 2 * np.pi * r
        return f
    
    def integrand(params):
        r1, r2, psi2 = params
        # integrand of the other term in the expression of SCov(-,-)
        x12 = r1 - r2 * np.cos(psi2)
        y12 = - r2 * np.sin(psi2)
        r12 = (x12**2 + y12**2)**0.5
        psi12 = np.arctan2(y12, x12)
        f = (3 * xi32_eps_plus_intp(r1) * xi32_eps_minus_intp(r2) * kill
             + xi1_eps_plus_intp(0) * xi2_eps_minus_intp(r12) * np.cos(4 * (psi12 - psi2))
            ) * 2 * np.pi * r1 * r2
        return f
    
    for a1 in range(Nbin):
        
        # term proportional to the Krönecker delta
        integral, err = monte_carlo_integrate(integrand_diag, [(rs[a1], rs[a1+1])])
        integral /= Omegas[a1]
        
        scov_pm[a1, a1] = (integral
                           - 2 * xi2_plus_bins[a1] * xi2_minus_bins[a1] * kill #maybe not
                           ) * Omegatot / Omegas[a1] / Nlens**2
        
        # term without the Krönecker delta
        for a2 in range(a1, Nbin):
            
            integral, err = monte_carlo_integrate(integrand,
                        [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1]), (0, 2*np.pi)])
            integral /= (Omegas[a1] * Omegas[a2]) # normalise
            
            scov_pm[a1, a2] += (integral
                                - 2 * xi2_plus_bins[a1] * xi2_minus_bins[a2] * kill # maybe not
                                ) * 2 / Nlens
            scov_pm[a2, a1] = scov_pm[a1, a2]
            
    
    # Make the full sparsity covariance matrix, with first xi_+ and then xi_-
    scov_mp = np.transpose(scov_pm)
    scov = np.block([[scov_pp, scov_pm],
                     [scov_mp, scov_mm]])
    
    
    return scov, scov_pp, scov_mm, scov_pm, scov_mp