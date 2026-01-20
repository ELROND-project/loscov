import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

from numba import njit

get_item('LLp','LLx','LP', 'PP', 'angular_distributions', 'redshift_distributions', 'L0')


################################################## JIT-compiled integrands ##############################################################

@njit
def _ccov_integrand_LPLP(params, r_grid, LLp_grid, LLx_grid, LP_B_grid, LP_D_grid, PP_BD_grid):
    """
    JIT-compiled cosmic covariance integrand for LPLP.
    Returns a single value (not multiple components like other files).
    """
    psi_b, psi_kd, r_b, r_kd, r_k = params

    # Geometry
    y_kb = r_b * np.sin(psi_b)
    x_kb = r_b * np.cos(psi_b) - r_k

    r_kb = np.sqrt(y_kb**2 + x_kb**2)
    psi_kb = np.arctan2(y_kb, x_kb)

    r_bd = cos_law_side_jit(r_kd, r_kb, (psi_kd - psi_kb))
    psi_bd = cos_law_angle_jit(r_kd, r_bd, r_kb) + psi_kd

    # Interpolate correlation function values
    LLp_rk = interp_jit(r_k, r_grid, LLp_grid)
    LLx_rk = interp_jit(r_k, r_grid, LLx_grid)
    LP_B_rk = interp_jit(r_k, r_grid, LP_B_grid)
    LP_D_rbd = interp_jit(r_bd, r_grid, LP_D_grid)
    PP_rbd = interp_jit(r_bd, r_grid, PP_BD_grid)

    # Pre-compute trig functions
    c2_b = cos2_jit(psi_b)
    s2_b = sin2_jit(psi_b)
    c2_kd = cos2_jit(psi_kd)
    s2_kd = sin2_jit(psi_kd)
    c2_bd_b = cos2_jit(psi_bd - psi_b)

    f = ((LLp_rk * c2_b * c2_kd + LLx_rk * s2_b * s2_kd) * PP_rbd
         + LP_D_rbd * LP_B_rk * c2_bd_b * c2_kd)

    jacobian = 2 * np.pi * r_k * r_b * r_kd

    return f * jacobian


@njit
def _ncov_integrand_LPLP_BD_equal(params, r_grid, LLp_grid, LLx_grid, PP_BD_grid):
    """
    JIT-compiled noise covariance integrand for LPLP when B == D.
    Computes both L and P components with shared geometry calculation.
    """
    r_i, r_k, psi_k = params

    # Geometry (computed once for both components)
    y_ik = r_k * np.sin(psi_k)
    x_ik = r_k * np.cos(psi_k) - r_i

    r_ik = np.sqrt(y_ik**2 + x_ik**2)
    psi_ik = np.arctan2(y_ik, x_ik)

    # Pre-compute trig functions
    c2_k = cos2_jit(psi_k)
    c2_ik = cos2_jit(psi_ik)
    s2_ik = sin2_jit(psi_ik)
    diff_ik_k = psi_ik - psi_k
    c2_ik_k = cos2_jit(diff_ik_k)
    s2_ik_k = sin2_jit(diff_ik_k)

    # Interpolate correlation function values
    LLp_rik = interp_jit(r_ik, r_grid, LLp_grid)
    LLx_rik = interp_jit(r_ik, r_grid, LLx_grid)
    PP_rik = interp_jit(r_ik, r_grid, PP_BD_grid)

    # Jacobian
    jacobian = 2 * np.pi * r_i * r_k

    # Compute both components
    f_L = 0.5 * PP_rik * c2_k
    f_P = LLp_rik * c2_ik * c2_ik_k + LLx_rik * s2_ik * s2_ik_k

    n = len(r_i)
    result = np.empty((2, n))
    result[0] = f_L * jacobian
    result[1] = f_P * jacobian
    return result


@njit
def _ncov_integrand_LPLP_BD_diff(params, r_grid, PP_BD_grid):
    """
    JIT-compiled noise covariance integrand for LPLP when B != D.
    Returns a single value (L component only).
    """
    r_i, r_k, psi_k = params

    # Geometry
    y_ik = r_k * np.sin(psi_k)
    x_ik = r_k * np.cos(psi_k) - r_i

    r_ik = np.sqrt(y_ik**2 + x_ik**2)

    # Pre-compute trig function
    c2_k = cos2_jit(psi_k)

    # Interpolate correlation function value
    PP_rik = interp_jit(r_ik, r_grid, PP_BD_grid)

    f = 0.5 * PP_rik * c2_k

    jacobian = 2 * np.pi * r_i * r_k

    return f * jacobian


################################################## LPLP cosmic covariance ##############################################################

def generate_ccov_LPLP(B, D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy position correlation functions.
    
    B             : the galaxy redshift bin B  (0 to Nbinz_P)
    D             : the galaxy redshift bin D (0 to Nbinz_P)
    """

    # Pre-compute grids for fast JIT-compiled interpolation
    n_grid_points = 2000
    r_grid, LLp_grid = spline_to_grid(LLp, 0, r2_max, n_points=n_grid_points)
    _, LLx_grid = spline_to_grid(LLx, 0, r2_max, n_points=n_grid_points)
    _, LP_B_grid = spline_to_grid(LP[B], 0, r2_max, n_points=n_grid_points)
    _, LP_D_grid = spline_to_grid(LP[D], 0, r2_max, n_points=n_grid_points)
    _, PP_BD_grid = spline_to_grid(PP[B][D], 0, r2_max, n_points=n_grid_points)

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
    
    # Wrapper for JIT-compiled integrand
    def integrand(params):
        """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
        return _ccov_integrand_LPLP(params, r_grid, LLp_grid, LLx_grid, LP_B_grid, LP_D_grid, PP_BD_grid)
    
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

    # Pre-compute grids for fast JIT-compiled interpolation
    n_grid_points = 2000
    r_grid, PP_BD_grid = spline_to_grid(PP[B][D], 0, r2_max, n_points=n_grid_points)
    if B == D:
        _, LLp_grid = spline_to_grid(LLp, 0, r2_max, n_points=n_grid_points)
        _, LLx_grid = spline_to_grid(LLx, 0, r2_max, n_points=n_grid_points)
    else:
        LLp_grid = None
        LLx_grid = None

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
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            return _ncov_integrand_LPLP_BD_equal(params, r_grid, LLp_grid, LLx_grid, PP_BD_grid)

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
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            return _ncov_integrand_LPLP_BD_diff(params, r_grid, PP_BD_grid)

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
