import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

from numba import njit

get_item('LLp', 'LLx', 'angular_distributions', 'L0')


################################################## JIT-compiled integrands ##############################################################

@njit
def _ccov_integrand_LLLL(params, r_grid, LLp_grid, LLx_grid):
    """
    JIT-compiled cosmic covariance integrand for LLLL.
    Computes pp, px, xp, xx components with shared geometry calculation.
    """
    psi_j, psi_kl, r_j, r_kl, r_k = params

    # Geometry (computed once for all 4 components)
    y_kj = r_j * np.sin(psi_j)
    x_kj = r_j * np.cos(psi_j) - r_k

    r_kj = np.sqrt(y_kj**2 + x_kj**2)
    psi_kj = np.arctan2(y_kj, x_kj)

    r_jl = cos_law_side_jit(r_kl, r_kj, (psi_kl - psi_kj))
    psi_jl = cos_law_angle_jit(r_kl, r_jl, r_kj) + psi_kl

    # Pre-compute trig functions (used multiple times)
    c2_j = cos2_jit(psi_j)
    s2_j = sin2_jit(psi_j)
    c2_kl = cos2_jit(psi_kl)
    s2_kl = sin2_jit(psi_kl)

    psi_jl_j = psi_jl - psi_j
    psi_jl_kl = psi_jl - psi_kl
    c2_jl_j = cos2_jit(psi_jl_j)
    s2_jl_j = sin2_jit(psi_jl_j)
    c2_jl_kl = cos2_jit(psi_jl_kl)
    s2_jl_kl = sin2_jit(psi_jl_kl)

    # Interpolate correlation function values (fast grid lookup)
    LLp_rk = interp_jit(r_k, r_grid, LLp_grid)
    LLx_rk = interp_jit(r_k, r_grid, LLx_grid)
    LLp_rjl = interp_jit(r_jl, r_grid, LLp_grid)
    LLx_rjl = interp_jit(r_jl, r_grid, LLx_grid)

    # Common jacobian factor
    jacobian = 2 * np.pi * r_k * r_j * r_kl

    # pp component
    f_pp = ((LLp_rk * c2_j * c2_kl + LLx_rk * s2_j * s2_kl)
          * (LLp_rjl * c2_jl_j * c2_jl_kl + LLx_rjl * s2_jl_j * s2_jl_kl))

    # px component
    f_px = -((LLp_rk * c2_j * s2_kl - LLx_rk * s2_j * c2_kl)
           * (LLp_rjl * c2_jl_j * s2_jl_kl - LLx_rjl * s2_jl_j * c2_jl_kl))

    # xp component
    f_xp = -((LLp_rk * s2_j * c2_kl - LLx_rk * c2_j * s2_kl)
           * (LLp_rjl * s2_jl_j * c2_jl_kl - LLx_rjl * c2_jl_j * s2_jl_kl))

    # xx component
    f_xx = ((LLp_rk * s2_j * s2_kl + LLx_rk * c2_j * c2_kl)
          * (LLp_rjl * s2_jl_j * s2_jl_kl + LLx_rjl * c2_jl_j * c2_jl_kl))

    n = len(r_k)
    result = np.empty((4, n))
    result[0] = f_pp * jacobian
    result[1] = f_px * jacobian
    result[2] = f_xp * jacobian
    result[3] = f_xx * jacobian
    return result


@njit
def _ncov_integrand_LLLL(params, r_grid, LLp_grid, LLx_grid):
    """
    JIT-compiled noise covariance integrand for LLLL.
    Computes pp, px, xp, xx components with shared geometry calculation.
    """
    r_j, r_l, psi_l = params

    # Geometry (computed once for all 4 components)
    y_jl = r_l * np.sin(psi_l)
    x_jl = r_l * np.cos(psi_l) - r_j

    r_jl = np.sqrt(y_jl**2 + x_jl**2)
    psi_jl = np.arctan2(y_jl, x_jl)

    # Pre-compute trig functions (used multiple times)
    c2_l = cos2_jit(psi_l)
    s2_l = sin2_jit(psi_l)
    c2_jl = cos2_jit(psi_jl)
    s2_jl = sin2_jit(psi_jl)

    psi_jl_l = psi_jl - psi_l
    c2_jl_l = cos2_jit(psi_jl_l)
    s2_jl_l = sin2_jit(psi_jl_l)

    # Interpolate correlation function values (fast grid lookup)
    LLp_rjl = interp_jit(r_jl, r_grid, LLp_grid)
    LLx_rjl = interp_jit(r_jl, r_grid, LLx_grid)

    # Common jacobian factor
    jacobian = 2 * np.pi * r_j * r_l

    # pp component
    f_pp = c2_l * (LLp_rjl * c2_jl * c2_jl_l + LLx_rjl * s2_jl * s2_jl_l)

    # px component
    f_px = -s2_l * (LLp_rjl * c2_jl * s2_jl_l - LLx_rjl * s2_jl * c2_jl_l)

    # xp component
    f_xp = s2_l * (LLp_rjl * s2_jl * c2_jl_l - LLx_rjl * c2_jl * s2_jl_l)

    # xx component
    f_xx = c2_l * (LLp_rjl * s2_jl * s2_jl_l + LLx_rjl * c2_jl * c2_jl_l)

    n = len(r_j)
    result = np.empty((4, n))
    result[0] = f_pp * jacobian
    result[1] = f_px * jacobian
    result[2] = f_xp * jacobian
    result[3] = f_xx * jacobian
    return result

################################################## LLLL cosmic covariance ##############################################################

def generate_ccov_LLLL():
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear correlation functions.
    """

    # Pre-compute grids for fast JIT-compiled interpolation
    r_grid_max = min(3 * r2_max, Thetamax)
    r_grid, LLp_grid = spline_to_grid(LLp, 0, r_grid_max, n_points=2000)
    _, LLx_grid = spline_to_grid(LLx, 0, r_grid_max, n_points=2000)

    def generate_matrices(sign1, sign2):

        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LL_{sign2}']

        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad)

        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LL
        Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LL (in rad)

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
        def integrand_all(params):
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            result = _ccov_integrand_LLLL(params, r_grid, LLp_grid, LLx_grid)
            return result

        def integral_bins(alpha, beta):
            """Compute all 4 component integrals with shared samples."""

            ranges = [(0, 2*np.pi), (0, 2*np.pi),
                      (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]

            integrals, errs = monte_carlo_integrate(integrand_all, ranges, Csamp)

            # normalisation of differential elements
            norm = 2 / (Omegatot * Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]

            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)

                ccov_pp[alpha, beta], ccov_px[alpha, beta], ccov_xp[alpha, beta], ccov_xx[alpha, beta] = integrals
                err_pp[alpha, beta], err_px[alpha, beta], err_xp[alpha, beta], err_xx[alpha, beta] = errs

                test_err(err_pp[alpha, beta], ccov_pp[alpha, beta], f'LLLL ccov plus plus angular bins {alpha, beta}')
                test_err(err_px[alpha, beta], ccov_px[alpha, beta], f'LLLL ccov plus times angular bins {alpha, beta}')
                test_err(err_xp[alpha, beta], ccov_xp[alpha, beta], f'LLLL ccov times plus angular bins {alpha, beta}')
                test_err(err_xx[alpha, beta], ccov_xx[alpha, beta], f'LLLL ccov times times angular bins {alpha, beta}')

        err = np.sqrt(err_pp**2 + err_px**2 + err_xp**2 + err_xx**2)

        return ccov_pp, ccov_px, ccov_xp, ccov_xx, err 

    #ccov_pp

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errpp = generate_matrices('plus', 'plus')

    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx

    #ccov_pm

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errpm = generate_matrices('plus', 'minus')

    ccovpm = ccov_pp - ccov_px + ccov_xp - ccov_xx

    #ccov_mp: use transpose relationship Cov(ξ-_α, ξ+_β) = Cov(ξ+_β, ξ-_α)
    # This avoids computing generate_matrices('minus', 'plus')

    ccovmp = ccovpm.T
    errmp = errpm.T

    #ccov_mm

    ccov_pp, ccov_px, ccov_xp, ccov_xx, errmm = generate_matrices('minus', 'minus')

    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx

    ccov = np.block([[ccovmm, ccovmp],
                     [ccovpm, ccovpp]])

    err = np.block([[errmm, errmp],
                     [errpm, errpp]])
    
    return ccov, err

################################################## 6.2.2 LeLe noise/sparsity covariance #############################################################

def generate_ncov_LLLL():
    """
    Computes the contribution of noise and sparsity variance in the
    covariance matrix of the LOS shear correlation functions.
    """

    # Pre-compute grids for fast JIT-compiled interpolation
    r_grid_max = min(3 * r2_max, Thetamax)
    r_grid, LLp_grid = spline_to_grid(LLp, 0, r_grid_max, n_points=2000)
    _, LLx_grid = spline_to_grid(LLx, 0, r_grid_max, n_points=2000)

    def generate_matrices(sign1, sign2):

        angular_distribution1 = angular_distributions[f'LL_{sign1}']
        angular_distribution2 = angular_distributions[f'LL_{sign2}']

        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LL (sign1)
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign1)
        rs1         = angular_distribution1.limits     #the angular bin limits for LL (in rad) (sign1)

        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LL (sign2)
        Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LL (in rad) (sign2)

        # Initialise the blocks
        ncov_pp = np.zeros((Nbin1, Nbin2))
        ncov_px = np.zeros((Nbin1, Nbin2))
        ncov_xp = np.zeros((Nbin1, Nbin2))
        ncov_xx = np.zeros((Nbin1, Nbin2))

        nerr_pp = np.zeros((Nbin1, Nbin2))
        nerr_px = np.zeros((Nbin1, Nbin2))
        nerr_xp = np.zeros((Nbin1, Nbin2))
        nerr_xx = np.zeros((Nbin1, Nbin2))

        # Initialise the blocks
        scov_pp = np.zeros((Nbin1, Nbin2))
        scov_px = np.zeros((Nbin1, Nbin2))
        scov_xp = np.zeros((Nbin1, Nbin2))
        scov_xx = np.zeros((Nbin1, Nbin2))

        serr_pp = np.zeros((Nbin1, Nbin2))
        serr_px = np.zeros((Nbin1, Nbin2))
        serr_xp = np.zeros((Nbin1, Nbin2))
        serr_xx = np.zeros((Nbin1, Nbin2))

        # Wrapper for JIT-compiled integrand
        def integrand_all(params):
            """Wrapper that calls JIT-compiled integrand with pre-computed grids."""
            result = _ncov_integrand_LLLL(params, r_grid, LLp_grid, LLx_grid)
            return result

        def integral_bins(alpha, beta):
            """Compute all 4 component integrals with shared samples."""

            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]

            integrals, errs = monte_carlo_integrate(integrand_all, ranges, Nsamp)

            # normalisation of differential elements
            norm = 2 / (Omegas1[alpha] * Omegas2[beta])
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
                
                #addition of constant term
                if alpha == beta:

                    cterm_n = (1/2) * ( (sigma_L**4+2*L0*sigma_L**2)/(Nlens**2) ) *  Omegatot / Omegas1[alpha]
                    cterm_s = (1/2) * ( (L0/Nlens)**2 ) * Omegatot / Omegas1[alpha]
                    
                    ncov_pp[alpha, beta] += cterm_n
                    ncov_xx[alpha, beta] += cterm_n
                    
                    scov_pp[alpha, beta] += cterm_s
                    scov_xx[alpha, beta] += cterm_s
    
                test_err(nerr_pp[alpha, beta], ncov_pp[alpha, beta], f'LLLL ncov plus plus angular bins {alpha, beta}')
                test_err(nerr_px[alpha, beta], ncov_px[alpha, beta], f'LLLL ncov plus times angular bins {alpha, beta}')
                test_err(nerr_xp[alpha, beta], ncov_xp[alpha, beta], f'LLLL ncov times plus angular bins {alpha, beta}')
                test_err(nerr_xx[alpha, beta], ncov_xx[alpha, beta], f'LLLL ncov times times angular bins {alpha, beta}')
    
                test_err(serr_pp[alpha, beta], scov_pp[alpha, beta], f'LLLL scov plus plus angular bins {alpha, beta}')
                test_err(serr_px[alpha, beta], scov_px[alpha, beta], f'LLLL scov plus times angular bins {alpha, beta}')
                test_err(serr_xp[alpha, beta], scov_xp[alpha, beta], f'LLLL scov times plus angular bins {alpha, beta}')
                test_err(serr_xx[alpha, beta], scov_xx[alpha, beta], f'LLLL scov times times angular bins {alpha, beta}')

        nerr = np.sqrt(nerr_pp**2 + nerr_px**2 + nerr_xp**2 + nerr_xx**2)
        serr = np.sqrt(serr_pp**2 + serr_px**2 + serr_xp**2 + serr_xx**2)

        return ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerr, serr    
    
    #plus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrpp, serrpp = generate_matrices('plus', 'plus')

    ncovpp = ncov_pp + ncov_px + ncov_xp + ncov_xx
    scovpp = scov_pp + scov_px + scov_xp + scov_xx

    #plus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx, nerrpm, serrpm = generate_matrices('plus', 'minus')

    ncovpm = ncov_pp - ncov_px + ncov_xp - ncov_xx
    scovpm = scov_pp - scov_px + scov_xp - scov_xx

    #minus plus: use transpose relationship Cov(ξ-_α, ξ+_β) = Cov(ξ+_β, ξ-_α)
    # This avoids computing generate_matrices('minus', 'plus')

    ncovmp = ncovpm.T
    scovmp = scovpm.T
    nerrmp = nerrpm.T
    serrmp = serrpm.T

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
