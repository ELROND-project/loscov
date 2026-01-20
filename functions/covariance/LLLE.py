import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

from numba import njit

get_item('LLp','LLx', 'LEp', 'LEx', 'angular_distributions', 'redshift_distributions', 'L0')


################################################## JIT-compiled integrands ##############################################################

@njit
def _ccov_integrand_LLLE(params, r_grid, LLp_grid, LLx_grid, LEp_D_grid, LEx_D_grid):
    """
    JIT-compiled cosmic covariance integrand for LLLE.
    Computes pp, px, xp, xx components with shared geometry calculation.
    """
    psi_j, psi_kd, r_j, r_kd, r_k = params

    # Geometry (computed once for all 4 components)
    y_kj = r_j * np.sin(psi_j)
    x_kj = r_j * np.cos(psi_j) - r_k

    r_kj = np.sqrt(y_kj**2 + x_kj**2)
    psi_kj = np.arctan2(y_kj, x_kj)

    r_jd = cos_law_side_jit(r_kd, r_kj, (psi_kd - psi_kj))
    psi_jd = cos_law_angle_jit(r_kd, r_jd, r_kj) + psi_kd

    # Pre-compute trig functions (used multiple times)
    c2_j = cos2_jit(psi_j)
    s2_j = sin2_jit(psi_j)
    c2_kd = cos2_jit(psi_kd)
    s2_kd = sin2_jit(psi_kd)
    diff_jd_j = psi_jd - psi_j
    diff_jd_kd = psi_jd - psi_kd
    c2_jd_j = cos2_jit(diff_jd_j)
    s2_jd_j = sin2_jit(diff_jd_j)
    c2_jd_kd = cos2_jit(diff_jd_kd)
    s2_jd_kd = sin2_jit(diff_jd_kd)

    # Interpolate correlation function values (fast grid lookup)
    idx_rk, t_rk = interp_index_weight_jit(r_k, r_grid)
    idx_rjd, t_rjd = interp_index_weight_jit(r_jd, r_grid)

    LLp_rk = interp_eval_jit(idx_rk, t_rk, LLp_grid)
    LLx_rk = interp_eval_jit(idx_rk, t_rk, LLx_grid)
    LEp_rjd = interp_eval_jit(idx_rjd, t_rjd, LEp_D_grid)
    LEx_rjd = interp_eval_jit(idx_rjd, t_rjd, LEx_D_grid)

    # Jacobian
    jacobian = 2 * np.pi * r_k * r_j * r_kd

    # Compute all 4 components
    f_pp = ((LLp_rk * c2_j * c2_kd + LLx_rk * s2_j * s2_kd)
            * (LEp_rjd * c2_jd_j * c2_jd_kd + LEx_rjd * s2_jd_j * s2_jd_kd))

    f_px = -((LLp_rk * c2_j * s2_kd - LLx_rk * s2_j * c2_kd)
             * (LEp_rjd * c2_jd_j * s2_jd_kd - LEx_rjd * s2_jd_j * c2_jd_kd))

    f_xp = -((LLp_rk * s2_j * c2_kd - LLx_rk * c2_j * s2_kd)
             * (LEp_rjd * s2_jd_j * c2_jd_kd - LEx_rjd * c2_jd_j * s2_jd_kd))

    f_xx = ((LLp_rk * s2_j * s2_kd + LLx_rk * c2_j * c2_kd)
            * (LEp_rjd * s2_jd_j * s2_jd_kd + LEx_rjd * c2_jd_j * c2_jd_kd))

    n = len(r_k)
    result = np.empty((4, n))
    result[0] = f_pp * jacobian
    result[1] = f_px * jacobian
    result[2] = f_xp * jacobian
    result[3] = f_xx * jacobian
    return result


@njit
def _ncov_integrand_LLLE(params, r_grid, LEp_D_grid, LEx_D_grid):
    """
    JIT-compiled noise covariance integrand for LLLE.
    """
    r_j, r_d, psi_d = params

    # Geometry
    y_jd = r_d * np.sin(psi_d)
    x_jd = r_d * np.cos(psi_d) - r_j

    r_jd = np.sqrt(y_jd**2 + x_jd**2)
    psi_jd = np.arctan2(y_jd, x_jd)

    # Pre-compute trig functions
    c2_d = cos2_jit(psi_d)
    s2_d = sin2_jit(psi_d)
    c2_jd = cos2_jit(psi_jd)
    s2_jd = sin2_jit(psi_jd)
    diff_jd_d = psi_jd - psi_d
    c2_jd_d = cos2_jit(diff_jd_d)
    s2_jd_d = sin2_jit(diff_jd_d)

    # Interpolate correlation function values
    idx_rjd, t_rjd = interp_index_weight_jit(r_jd, r_grid)
    LEp_rjd = interp_eval_jit(idx_rjd, t_rjd, LEp_D_grid)
    LEx_rjd = interp_eval_jit(idx_rjd, t_rjd, LEx_D_grid)

    # Jacobian
    jacobian = 2 * np.pi * r_j * r_d

    # Compute all 4 components
    f_pp = c2_d * (LEp_rjd * c2_jd * c2_jd_d + LEx_rjd * s2_jd * s2_jd_d)
    f_px = -s2_d * (LEp_rjd * c2_jd * s2_jd_d - LEx_rjd * s2_jd * c2_jd_d)
    f_xp = s2_d * (LEp_rjd * s2_jd * c2_jd_d - LEx_rjd * c2_jd * s2_jd_d)
    f_xx = c2_d * (LEp_rjd * s2_jd * s2_jd_d + LEx_rjd * c2_jd * c2_jd_d)

    n = len(r_j)
    result = np.empty((4, n))
    result[0] = f_pp * jacobian
    result[1] = f_px * jacobian
    result[2] = f_xp * jacobian
    result[3] = f_xx * jacobian
    return result


################################################## LLLE cosmic covariance ##############################################################

def generate_ccov_LLLE(D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - LOS shear x LOS shear - galaxy shape correlation functions.

    D             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    # Pre-compute grids for fast JIT-compiled interpolation (done once for all sign combinations)
    n_grid_points = 2000
    r_grid_max = min(3 * r2_max, Thetamax)
    r_grid, LLp_grid = spline_to_grid(LLp, 0, r_grid_max, n_points=n_grid_points)
    _, LLx_grid = spline_to_grid(LLx, 0, r_grid_max, n_points=n_grid_points)
    _, LEp_D_grid = spline_to_grid(LEp[D], 0, r_grid_max, n_points=n_grid_points)
    _, LEx_D_grid = spline_to_grid(LEx[D], 0, r_grid_max, n_points=n_grid_points)

    def generate_matrices(sign1, sign2):

        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LE_{sign2}'][D]

        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)

        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LE
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LE (in rad)

        # Initialise the blocks
        ccov_pp = np.zeros((Nbin1, Nbin2))
        ccov_px = np.zeros((Nbin1, Nbin2))
        ccov_xp = np.zeros((Nbin1, Nbin2))
        ccov_xx = np.zeros((Nbin1, Nbin2))

        err_pp = np.zeros((Nbin1, Nbin2))
        err_px = np.zeros((Nbin1, Nbin2))
        err_xp = np.zeros((Nbin1, Nbin2))
        err_xx = np.zeros((Nbin1, Nbin2))

        # Wrapper for JIT-compiled integrand
        @njit
        def integrand_all(params):
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            return _ccov_integrand_LLLE(params, r_grid, LLp_grid, LLx_grid, LEp_D_grid, LEx_D_grid)

        def integral_bins(alpha, beta):
            """Compute all 4 component integrals with shared samples."""
            ranges = [(0, 2*np.pi), (0, 2*np.pi),
                      (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]

            integrals, errs = monte_carlo_integrate_jit(integrand_all, ranges, Csamp)

            # normalisation of differential elements
            norm = 2/(Omegatot * Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]
            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)
                ccov_pp[alpha, beta], ccov_px[alpha, beta], ccov_xp[alpha, beta], ccov_xx[alpha, beta] = integrals
                err_pp[alpha, beta], err_px[alpha, beta], err_xp[alpha, beta], err_xx[alpha, beta] = errs
    
                test_err(err_pp[alpha, beta], ccov_pp[alpha, beta], f'LLLE ccov plus plus angular bins {alpha, beta}')
                test_err(err_px[alpha, beta], ccov_px[alpha, beta], f'LLLE ccov plus times angular bins {alpha, beta}')
                test_err(err_xp[alpha, beta], ccov_xp[alpha, beta], f'LLLE ccov times plus angular bins {alpha, beta}')
                test_err(err_xx[alpha, beta], ccov_xx[alpha, beta], f'LLLE ccov times times angular bins {alpha, beta}')

        err = np.sqrt(err_pp**2 + err_px**2 + err_xp**2 + err_xx**2)

        return ccov_pp, ccov_px, ccov_xp, ccov_xx, err

    #plus plus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errpp = generate_matrices('plus', 'plus')
    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx
    
    #plus minus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errpm = generate_matrices('plus', 'minus')
    ccovpm = ccov_pp - ccov_px + ccov_xp - ccov_xx

    #minus plus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errmp = generate_matrices('minus', 'plus')
    ccovmp = ccov_pp + ccov_px - ccov_xp - ccov_xx

    #minus minus

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errmm = generate_matrices('minus', 'minus')
    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx
    
    ccov = np.block([[ccovmm, ccovmp],
                     [ccovpm, ccovpp]])
    
    err = np.block([[errmm, errmp],
                     [errpm, errpp]])
    
    return ccov, err

################################################## LLLE noise/sparsity covariance #############################################################

def generate_ncov_LLLE(D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear correlation functions.

    D             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    # Pre-compute grids for fast JIT-compiled interpolation (done once for all sign combinations)
    n_grid_points = 2000
    r_grid_max = min(3 * r2_max, Thetamax)
    r_grid, LEp_D_grid = spline_to_grid(LEp[D], 0, r_grid_max, n_points=n_grid_points)
    _, LEx_D_grid = spline_to_grid(LEx[D], 0, r_grid_max, n_points=n_grid_points)

    def generate_matrices(sign1, sign2):

        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LE_{sign2}'][D]

        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)

        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LE
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LE (in rad)

        # Initialise the blocks
        ncov_pp = np.zeros((Nbin1, Nbin2))
        ncov_px = np.zeros((Nbin1, Nbin2))
        ncov_xp = np.zeros((Nbin1, Nbin2))
        ncov_xx = np.zeros((Nbin1, Nbin2))

        nerr_pp = np.zeros((Nbin1, Nbin2))
        nerr_px = np.zeros((Nbin1, Nbin2))
        nerr_xp = np.zeros((Nbin1, Nbin2))
        nerr_xx = np.zeros((Nbin1, Nbin2))

        scov_pp = np.zeros((Nbin1, Nbin2))
        scov_px = np.zeros((Nbin1, Nbin2))
        scov_xp = np.zeros((Nbin1, Nbin2))
        scov_xx = np.zeros((Nbin1, Nbin2))

        serr_pp = np.zeros((Nbin1, Nbin2))
        serr_px = np.zeros((Nbin1, Nbin2))
        serr_xp = np.zeros((Nbin1, Nbin2))
        serr_xx = np.zeros((Nbin1, Nbin2))

        # Wrapper for JIT-compiled integrand
        @njit
        def integrand_all(params):
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            return _ncov_integrand_LLLE(params, r_grid, LEp_D_grid, LEx_D_grid)

        def integral_bins(alpha, beta):
            """Compute all 4 component integrals with shared samples."""
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]

            integrals, errs = monte_carlo_integrate_jit(integrand_all, ranges, Nsamp)

            # normalisation of differential elements
            norm = 1/(Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]
            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)
                int_pp, int_px, int_xp, int_xx = integrals
                err_pp, err_px, err_xp, err_xx = errs

                ncov_pp[alpha, beta] = (sigma_L**2/Nlens) * int_pp
                nerr_pp[alpha, beta] = (sigma_L**2/Nlens) * err_pp

                scov_pp[alpha, beta] = (L0/Nlens) * int_pp
                serr_pp[alpha, beta] = (L0/Nlens) * err_pp

                ncov_px[alpha, beta] = (sigma_L**2/Nlens) * int_px
                nerr_px[alpha, beta] = (sigma_L**2/Nlens) * err_px

                scov_px[alpha, beta] = (L0/Nlens) * int_px
                serr_px[alpha, beta] = (L0/Nlens) * err_px

                ncov_xp[alpha, beta] = (sigma_L**2/Nlens) * int_xp
                nerr_xp[alpha, beta] = (sigma_L**2/Nlens) * err_xp

                scov_xp[alpha, beta] = (L0/Nlens) * int_xp
                serr_xp[alpha, beta] = (L0/Nlens) * err_xp

                ncov_xx[alpha, beta] = (sigma_L**2/Nlens) * int_xx
                nerr_xx[alpha, beta] = (sigma_L**2/Nlens) * err_xx

                scov_xx[alpha, beta] = (L0/Nlens) * int_xx
                serr_xx[alpha, beta] = (L0/Nlens) * err_xx
    
                test_err(nerr_pp[alpha, beta], ncov_pp[alpha, beta], f'LLLE ncov plus plus angular bins {alpha, beta}')
                test_err(nerr_px[alpha, beta], ncov_px[alpha, beta], f'LLLE ncov plus times angular bins {alpha, beta}')
                test_err(nerr_xp[alpha, beta], ncov_xp[alpha, beta], f'LLLE ncov times plus angular bins {alpha, beta}')
                test_err(nerr_xx[alpha, beta], ncov_xx[alpha, beta], f'LLLE ncov times times angular bins {alpha, beta}')
    
                test_err(serr_pp[alpha, beta], scov_pp[alpha, beta], f'LLLE scov plus plus angular bins {alpha, beta}')
                test_err(serr_px[alpha, beta], scov_px[alpha, beta], f'LLLE scov plus times angular bins {alpha, beta}')
                test_err(serr_xp[alpha, beta], scov_xp[alpha, beta], f'LLLE scov times plus angular bins {alpha, beta}')
                test_err(serr_xx[alpha, beta], scov_xx[alpha, beta], f'LLLE scov times times angular bins {alpha, beta}')

        nerr = np.sqrt(nerr_pp**2 + nerr_px**2 + nerr_xp**2 + nerr_xx**2)
        serr = np.sqrt(serr_pp**2 + serr_px**2 + serr_xp**2 + serr_xx**2) 

        return ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerr, serr 
    
    #plus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrpp, serrpp  = generate_matrices('plus', 'plus')
    
    ncovpp = ncov_pp + ncov_px + ncov_xp + ncov_xx
    scovpp = scov_pp + scov_px + scov_xp + scov_xx
    
    #plus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrpm, serrpm = generate_matrices('plus', 'minus')
    
    ncovpm = ncov_pp - ncov_px + ncov_xp - ncov_xx
    scovpm = scov_pp - scov_px + scov_xp - scov_xx
    
    #minus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrmp, serrmp = generate_matrices('minus', 'plus')
    
    ncovmp = ncov_pp + ncov_px - ncov_xp - ncov_xx
    scovmp = scov_pp + scov_px - scov_xp - scov_xx
    
    #minus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrmm, serrmm = generate_matrices('minus', 'minus')
    
    ncovmm = ncov_pp - ncov_px - ncov_xp + ncov_xx
    scovmm = scov_pp - scov_px - scov_xp + scov_xx

    ncov = np.block([[ncovmm, ncovmp],
                     [ncovpm, ncovpp]])

    nerr = np.block([[nerrmm, nerrmp],
                     [nerrpm, nerrpp]])
    
    scov = np.block([[scovmm, scovmp],
                     [scovpm, scovpp]])

    serr = np.block([[serrmm, serrmp],
                     [serrpm, serrpp]])
    
    return [ncov, scov], [nerr, serr]
