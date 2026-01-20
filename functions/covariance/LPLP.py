import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLx','LP', 'PP', 'angular_distributions', 'redshift_distributions', 'L0')

################################################## LPLP cosmic covariance ##############################################################

def generate_ccov_LPLP(B, D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy position correlation functions.
    
    B             : the galaxy redshift bin B  (0 to Nbinz_P)
    D             : the galaxy redshift bin D (0 to Nbinz_P)
    """
    
    angular_distribution1 = angular_distributions['LP'][B]
    angular_distribution2 = angular_distributions['LP'][D]
    
    Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LP (redshift bin B)
    Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (redshift bin B)
    rs1         = angular_distribution1.limits     #the angular bin limits for LP (in rad) (redshift bin B)
    
    Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LP (redshift bin D) 
    Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (redshift bin D)
    rs2         = angular_distribution2.limits     #the angular bin limits for LP (in rad) (redshift bin D)
    
    # Initialise the blocks
    ccov = np.zeros((Nbin1, Nbin2))
    err = np.zeros((Nbin1, Nbin2))
    
    # Define the integrands (complete from here)
    
    def integrand(params):
        
        psi_b, psi_kd, r_b, r_kd, r_k = params
    
        y_kb = r_b*np.sin(psi_b)
        x_kb = r_b*np.cos(psi_b) - r_k
        
        r_kb = np.sqrt( y_kb**2 + x_kb**2 ) 
        psi_kb = np.arctan2(y_kb, x_kb)
        
        r_bd = cos_law_side(r_kd, r_kb, (psi_kd-psi_kb))
        psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd
        
        f = ( (LLp(r_k) * cos2(psi_b) * cos2(psi_kd)
            + LLx(r_k) * sin2(psi_b) * sin2(psi_kd))
            * PP[B][D](r_bd)
            + LP[D](r_bd) * LP[B](r_k) * cos2(psi_bd-psi_b) * cos2(psi_kd)
            )
        
        f *= 2 * np.pi * r_k * r_b * r_kd
        
        return f
    
    def integral_bins(integrand, alpha, beta):
        
        ranges = [(0, 2*np.pi), (0, 2*np.pi),
                  (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]
        
        integral, err = monte_carlo_integrate(integrand, ranges, Csamp)
        
        # normalisation of differential elements
        integral /= (Omegatot * Omegas1[alpha] * Omegas2[beta]) 
        err /= (Omegatot * Omegas1[alpha] * Omegas2[beta]) 
        return integral, err
    
    for alpha in range(Nbin1):
        for beta in range(Nbin2): 
                     
            ccov[alpha, beta], err[alpha, beta] = integral_bins(integrand, alpha, beta)

            test_err(err[alpha, beta], ccov[alpha, beta], f'LPLP ccov redshifts {B, D} angular bins {alpha, beta}')
            
    
    # Make the full cosmic covariance matrix

    ccov = np.block([[ccov]])
    err = np.block([[err]])
    
    return ccov, err

################################################## LPLP noise/sparsity covariance #############################################################

def generate_ncov_LPLP(B, D):
    """
    Computes the contribution of noise and sparsity variance in the 
    covariance matrix of the LOS shear - galaxy position correlation functions.
    
    B             : the galaxy redshift bin B  (0 to 4)
    D             : the galaxy redshift bin D (0 to 4)
    """
    
    angular_distribution1 = angular_distributions['LP'][B]
    angular_distribution2 = angular_distributions['LP'][D]
    
    Nbin1       = angular_distribution1.Nbina      #the number of angular bins
    Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = angular_distribution1.limits     #the angular bin limits (in rad)
    
    Nbin2       = angular_distribution2.Nbina      #the number of angular bins
    Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs2         = angular_distribution2.limits     #the angular bin limits (in rad)
    
    redshift_distribution = redshift_distributions['P']

    G_B    = redshift_distribution.get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
    
    # Initialise the blocks
    ncov = np.zeros((Nbin1, Nbin2))
    scov = np.zeros((Nbin1, Nbin2))
    
    nerr = np.zeros((Nbin1, Nbin2))
    serr = np.zeros((Nbin1, Nbin2))
    
    # Define combined integrand for both components (shares geometry and correlation evaluations)
    # When B == D, we compute both L and P components; otherwise only L

    if B == D:
        def integrand_all(params):
            """Compute L and P components with shared geometry calculation."""
            r_i, r_k, psi_k = params

            # Geometry (computed once for both components)
            y_ik = r_k*np.sin(psi_k)
            x_ik = r_k*np.cos(psi_k) - r_i

            r_ik = np.sqrt( y_ik**2 + x_ik**2 )
            psi_ik = np.arctan2(y_ik, x_ik)

            # Pre-compute trig functions (used multiple times)
            c2_k = cos2(psi_k)
            c2_ik = cos2(psi_ik)
            s2_ik = sin2(psi_ik)
            diff_ik_k = psi_ik - psi_k
            c2_ik_k = cos2(diff_ik_k)
            s2_ik_k = sin2(diff_ik_k)

            # Pre-compute correlation function values (expensive spline evaluations)
            LLp_rik = LLp(r_ik)
            LLx_rik = LLx(r_ik)
            PP_rik = PP[B][D](r_ik)

            # Jacobian
            jacobian = 2 * np.pi * r_i * r_k

            # Compute both components
            f_L = (1/2) * PP_rik * c2_k
            f_P = LLp_rik * c2_ik * c2_ik_k + LLx_rik * s2_ik * s2_ik_k

            return np.array([f_L * jacobian, f_P * jacobian])

        def integral_bins(alpha, beta):
            """Compute both component integrals with shared samples."""
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]

            integrals, errs = monte_carlo_integrate(integrand_all, ranges, Nsamp)

            # normalisation of differential elements
            norm = 1/(Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]
            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)
                int_L, int_P = integrals
                err_L, err_P = errs

                ncov[alpha, beta] = (sigma_L**2/Nlens) * int_L + (1/G_B) * int_P
                nerr[alpha, beta] = (sigma_L**2/Nlens) * err_L

                scov[alpha, beta] = (L0/Nlens) * int_L + (1/G_B) * int_P
                serr[alpha, beta] = (L0/Nlens) * err_L

                if alpha == beta:
                    ncov[alpha, beta] += (1/2) * (sigma_L**2 / (Nlens*G_B) ) * (Omegatot/Omegas1[alpha])
                    scov[alpha, beta] += (1/2) * (L0 / (Nlens*G_B) ) * (Omegatot/Omegas1[alpha])

                test_err(nerr[alpha, beta], ncov[alpha, beta], f'LPLP ncov redshifts {B, D} angular bins {alpha, beta}')
                test_err(serr[alpha, beta], scov[alpha, beta], f'LPLP scov redshifts {B, D} angular bins {alpha, beta}')

    else:
        # When B != D, only integrand_L is needed
        def integrand_L(params):
            r_i, r_k, psi_k = params

            y_ik = r_k*np.sin(psi_k)
            x_ik = r_k*np.cos(psi_k) - r_i

            r_ik = np.sqrt( y_ik**2 + x_ik**2 )

            f = (1/2) * PP[B][D](r_ik) * cos2(psi_k)

            f *= 2 * np.pi * r_i * r_k

            return f

        def integral_bins(alpha, beta):
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]

            integral, err = monte_carlo_integrate(integrand_L, ranges, Nsamp)

            # normalisation of differential elements
            norm = 1/(Omegas1[alpha] * Omegas2[beta])
            integral *= norm
            err *= norm
            return integral, err

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                int_L, err_L = integral_bins(alpha, beta)

                ncov[alpha, beta] = (sigma_L**2/Nlens) * int_L
                nerr[alpha, beta] = (sigma_L**2/Nlens) * err_L

                scov[alpha, beta] = (L0/Nlens) * int_L
                serr[alpha, beta] = (L0/Nlens) * err_L

                test_err(nerr[alpha, beta], ncov[alpha, beta], f'LPLP ncov redshifts {B, D} angular bins {alpha, beta}')
                test_err(serr[alpha, beta], scov[alpha, beta], f'LPLP scov redshifts {B, D} angular bins {alpha, beta}')

    # Make the full cosmic covariance matrix

    ncov = np.block([[ncov]])
    scov = np.block([[scov]])

    nerr = np.block([[nerr]])
    serr = np.block([[serr]])
    
    return [ncov, scov], [nerr, serr]