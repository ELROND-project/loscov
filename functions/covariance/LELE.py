import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLx', 'LEp', 'LEx', 'EEp', 'EEx', 'angular_distributions', 'redshift_distributions', 'L0', 'E0')

################################################## LELE cosmic covariance ##############################################################

def generate_ccov_LELE(B, D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy shape correlation functions.

    B             : the galaxy redshift bin D (0 to Nbinz_E)
    D             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    def generate_matrices(sign1, sign2):    

        angular_distribution1 = angular_distributions[f'LE_{sign1}'][B]
        angular_distribution2 = angular_distributions[f'LE_{sign2}'][D]
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LE 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LE (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LE 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
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
    
        # Define combined integrand that computes all 4 components with shared geometry

        def integrand_all(params):
            """Compute pp, px, xp, xx components with shared geometry calculation."""

            psi_b, psi_kd, r_b, r_kd, r_k = params

            # Geometry (computed once for all 4 components)
            y_kb = r_b * np.sin(psi_b)
            x_kb = r_b * np.cos(psi_b) - r_k

            r_kb = np.sqrt(y_kb**2 + x_kb**2)
            psi_kb = np.arctan2(y_kb, x_kb)

            r_bd = cos_law_side(r_kd, r_kb, (psi_kd - psi_kb))
            psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd

            # Pre-compute trig functions (used multiple times)
            c2_b = cos2(psi_b)
            s2_b = sin2(psi_b)
            c2_kd = cos2(psi_kd)
            s2_kd = sin2(psi_kd)

            psi_bd_b = psi_bd - psi_b
            psi_bd_kd = psi_bd - psi_kd
            c2_bd_b = cos2(psi_bd_b)
            s2_bd_b = sin2(psi_bd_b)
            c2_bd_kd = cos2(psi_bd_kd)
            s2_bd_kd = sin2(psi_bd_kd)

            # Pre-compute correlation function values (expensive spline evaluations)
            LLp_rk = LLp(r_k)
            LLx_rk = LLx(r_k)
            EEp_rbd = EEp[B][D](r_bd)
            EEx_rbd = EEx[B][D](r_bd)
            LEp_D_rk = LEp[D](r_k)
            LEx_D_rk = LEx[D](r_k)
            LEp_B_rbd = LEp[B](r_bd)
            LEx_B_rbd = LEx[B](r_bd)

            # Common jacobian factor
            jacobian = 2 * np.pi * r_k * r_b * r_kd

            # First factors for each component (LL and LE at r_k)
            LL_pp = LLp_rk * c2_b * c2_kd + LLx_rk * s2_b * s2_kd
            LL_px = LLp_rk * c2_b * s2_kd - LLx_rk * s2_b * c2_kd
            LL_xp = LLp_rk * s2_b * c2_kd - LLx_rk * c2_b * s2_kd
            LL_xx = LLp_rk * s2_b * s2_kd + LLx_rk * c2_b * c2_kd

            LE_D_pp = LEp_D_rk * c2_b * c2_kd + LEx_D_rk * s2_b * s2_kd
            LE_D_px = LEp_D_rk * c2_b * s2_kd - LEx_D_rk * s2_b * c2_kd
            LE_D_xp = LEp_D_rk * s2_b * c2_kd - LEx_D_rk * c2_b * s2_kd
            LE_D_xx = LEp_D_rk * s2_b * s2_kd + LEx_D_rk * c2_b * c2_kd

            # Second factors for each component (EE and LE at r_bd)
            EE_pp = EEp_rbd * c2_bd_b * c2_bd_kd + EEx_rbd * s2_bd_b * s2_bd_kd
            EE_px = EEp_rbd * c2_bd_b * s2_bd_kd - EEx_rbd * s2_bd_b * c2_bd_kd
            EE_xp = EEp_rbd * s2_bd_b * c2_bd_kd - EEx_rbd * c2_bd_b * s2_bd_kd
            EE_xx = EEp_rbd * s2_bd_b * s2_bd_kd + EEx_rbd * c2_bd_b * c2_bd_kd

            LE_B_pp = LEp_B_rbd * c2_bd_b * c2_bd_kd + LEx_B_rbd * s2_bd_b * s2_bd_kd
            LE_B_px = LEp_B_rbd * c2_bd_b * s2_bd_kd - LEx_B_rbd * s2_bd_b * c2_bd_kd
            LE_B_xp = LEp_B_rbd * s2_bd_b * c2_bd_kd - LEx_B_rbd * c2_bd_b * s2_bd_kd
            LE_B_xx = LEp_B_rbd * s2_bd_b * s2_bd_kd + LEx_B_rbd * c2_bd_b * c2_bd_kd

            # Combine components
            f_pp = (LL_pp * EE_pp + LE_D_pp * LE_B_pp)
            f_px = -(LL_px * EE_px + LE_D_px * LE_B_px)
            f_xp = -(LL_xp * EE_xp + LE_D_xp * LE_B_xp)
            f_xx = (LL_xx * EE_xx + LE_D_xx * LE_B_xx)

            return np.array([f_pp * jacobian, f_px * jacobian, f_xp * jacobian, f_xx * jacobian])

        def integral_bins(alpha, beta):
            """Compute all 4 component integrals with shared samples."""

            ranges = [(0, 2*np.pi), (0, 2*np.pi),
                      (rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, r2_max)]

            integrals, errs = monte_carlo_integrate(integrand_all, ranges, Csamp)

            # normalisation of differential elements
            norm = 1 / (Omegatot * Omegas1[alpha] * Omegas2[beta])
            integrals = [i * norm for i in integrals]
            errs = [e * norm for e in errs]

            return integrals, errs

        for alpha in range(Nbin1):
            for beta in range(Nbin2):

                integrals, errs = integral_bins(alpha, beta)

                ccov_pp[alpha, beta], ccov_px[alpha, beta], ccov_xp[alpha, beta], ccov_xx[alpha, beta] = integrals
                err_pp[alpha, beta], err_px[alpha, beta], err_xp[alpha, beta], err_xx[alpha, beta] = errs

                test_err(err_pp[alpha, beta], ccov_pp[alpha, beta], f'LELE ccov plus plus redshift bins{B, D} angular bins {alpha, beta}')
                test_err(err_px[alpha, beta], ccov_px[alpha, beta], f'LELE ccov plus times redshift bins{B, D} angular bins {alpha, beta}')
                test_err(err_xp[alpha, beta], ccov_xp[alpha, beta], f'LELE ccov times plus redshift bins{B, D} angular bins {alpha, beta}')
                test_err(err_xx[alpha, beta], ccov_xx[alpha, beta], f'LELE ccov times times redshift bins{B, D} angular bins {alpha, beta}')

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

################################################## LELE noise/sparsity covariance #############################################################

def generate_ncov_LELE(B, D):
    """
    Computes the contribution of noise and sparsity variance in the covariance matrix
    of the LOS shear - galaxy shape correlation functions.
    
    B             : the galaxy redshift bin D (0 to Nbinz_E)
    D             : the galaxy redshift bin D (0 to Nbinz_E)
    """

    def generate_matrices(sign1, sign2):  
    
        angular_distribution1 = angular_distributions[f'LE_{sign1}'][B]
        angular_distribution2 = angular_distributions[f'LE_{sign2}'][D]
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LE (sign1) 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign1)
        rs1         = angular_distribution1.limits     #the angular bin limits for LE (in rad) (sign1)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LE (sign2)
        Omegas2     = angular_distribution2.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2) (sign2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LE (in rad) (sign2)
        
        redshift_distribution = redshift_distributions['E']
        
        G_B    = redshift_distribution.get_ngal(B)         #G_B in the math - the number of galaxies in redshift bin B
        
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
        
        # Define combined integrands for L terms (always used) and E terms (used when B==D)

        def integrand_all_L(params):
            """Compute pp, px, xp, xx L-components with shared geometry."""

            r_b, r_d, psi_d = params

            # Geometry (computed once)
            y_bd = r_d * np.sin(psi_d)
            x_bd = r_d * np.cos(psi_d) - r_b

            r_bd = np.sqrt(y_bd**2 + x_bd**2)
            psi_bd = np.arctan2(y_bd, x_bd)

            # Pre-compute trig functions
            c2_d = cos2(psi_d)
            s2_d = sin2(psi_d)
            c2_bd = cos2(psi_bd)
            s2_bd = sin2(psi_bd)

            psi_bd_d = psi_bd - psi_d
            c2_bd_d = cos2(psi_bd_d)
            s2_bd_d = sin2(psi_bd_d)

            # Pre-compute correlation function values
            EEp_rbd = EEp[B][D](r_bd)
            EEx_rbd = EEx[B][D](r_bd)

            # Common jacobian factor (factor of 2 absorbed into 1/2 prefactor)
            jacobian = np.pi * r_b * r_d

            # Components using EE correlations
            f_pp = c2_d * (EEp_rbd * c2_bd * c2_bd_d + EEx_rbd * s2_bd * s2_bd_d)
            f_px = -s2_d * (EEp_rbd * c2_bd * s2_bd_d - EEx_rbd * s2_bd * c2_bd_d)
            f_xp = s2_d * (EEp_rbd * s2_bd * c2_bd_d - EEx_rbd * c2_bd * s2_bd_d)
            f_xx = c2_d * (EEp_rbd * s2_bd * s2_bd_d + EEx_rbd * c2_bd * c2_bd_d)

            return np.array([f_pp * jacobian, f_px * jacobian, f_xp * jacobian, f_xx * jacobian])

        def integrand_all_E(params):
            """Compute pp, px, xp, xx E-components with shared geometry (used when B==D)."""

            r_b, r_d, psi_d = params

            # Geometry (computed once)
            y_bd = r_d * np.sin(psi_d)
            x_bd = r_d * np.cos(psi_d) - r_b

            r_bd = np.sqrt(y_bd**2 + x_bd**2)
            psi_bd = np.arctan2(y_bd, x_bd)

            # Pre-compute trig functions
            c2_d = cos2(psi_d)
            s2_d = sin2(psi_d)
            c2_bd = cos2(psi_bd)
            s2_bd = sin2(psi_bd)

            psi_bd_d = psi_bd - psi_d
            c2_bd_d = cos2(psi_bd_d)
            s2_bd_d = sin2(psi_bd_d)

            # Pre-compute correlation function values
            LLp_rbd = LLp(r_bd)
            LLx_rbd = LLx(r_bd)

            # Common jacobian factor
            jacobian = np.pi * r_b * r_d

            # Components using LL correlations
            f_pp = c2_d * (LLp_rbd * c2_bd * c2_bd_d + LLx_rbd * s2_bd * s2_bd_d)
            f_px = -s2_d * (LLp_rbd * c2_bd * s2_bd_d - LLx_rbd * s2_bd * c2_bd_d)
            f_xp = s2_d * (LLp_rbd * s2_bd * c2_bd_d - LLx_rbd * c2_bd * s2_bd_d)
            f_xx = c2_d * (LLp_rbd * s2_bd * s2_bd_d + LLx_rbd * c2_bd * c2_bd_d)

            return np.array([f_pp * jacobian, f_px * jacobian, f_xp * jacobian, f_xx * jacobian])

        def integral_bins(alpha, beta):
            """Compute all component integrals with shared samples."""

            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]

            # Always compute L terms
            integrals_L, errs_L = monte_carlo_integrate(integrand_all_L, ranges, Nsamp)

            # Compute E terms only if B == D
            if B == D:
                integrals_E, errs_E = monte_carlo_integrate(integrand_all_E, ranges, Nsamp)
            else:
                integrals_E = [0, 0, 0, 0]
                errs_E = [0, 0, 0, 0]

            # Normalisation
            norm = 1 / (Omegas1[alpha] * Omegas2[beta])
            integrals_L = [i * norm for i in integrals_L]
            errs_L = [e * norm for e in errs_L]
            integrals_E = [i * norm for i in integrals_E]
            errs_E = [e * norm for e in errs_E]

            return integrals_L, errs_L, integrals_E, errs_E

        for alpha in range(Nbin1):

            for beta in range(Nbin2):

                integrals_L, errs_L, integrals_E, errs_E = integral_bins(alpha, beta)
                int_pp_L, int_px_L, int_xp_L, int_xx_L = integrals_L
                err_pp_L, err_px_L, err_xp_L, err_xx_L = errs_L
                int_pp_E, int_px_E, int_xp_E, int_xx_E = integrals_E
                err_pp_E, err_px_E, err_xp_E, err_xx_E = errs_E
    
                ncov_pp[alpha, beta] = (sigma_L**2/Nlens) * int_pp_L
                nerr_pp[alpha, beta] = (sigma_L**2/Nlens) * err_pp_L
                scov_pp[alpha, beta] = (L0/Nlens) * int_pp_L
                serr_pp[alpha, beta] = (L0/Nlens) * err_pp_L

                ncov_px[alpha, beta] = (sigma_L**2/Nlens) * int_px_L
                nerr_px[alpha, beta] = (sigma_L**2/Nlens) * err_px_L
                scov_px[alpha, beta] = (L0/Nlens) * int_px_L
                serr_px[alpha, beta] = (L0/Nlens) * err_px_L

                ncov_xp[alpha, beta] = (sigma_L**2/Nlens) * int_xp_L
                nerr_xp[alpha, beta] = (sigma_L**2/Nlens) * err_xp_L
                scov_xp[alpha, beta] = (L0/Nlens) * int_xp_L
                serr_xp[alpha, beta] = (L0/Nlens) * err_xp_L

                ncov_xx[alpha, beta] = (sigma_L**2/Nlens) * int_xx_L
                nerr_xx[alpha, beta] = (sigma_L**2/Nlens) * err_xx_L
                scov_xx[alpha, beta] = (L0/Nlens) * int_xx_L
                serr_xx[alpha, beta] = (L0/Nlens) * err_xx_L

                # Addition of E terms (only when B == D)
                if B == D:
                    ncov_pp[alpha, beta] += (sigma_E**2/G_B) * int_pp_E
                    nerr_pp[alpha, beta] = np.sqrt(nerr_pp[alpha, beta]**2 + ((sigma_E**2/G_B) * err_pp_E)**2)
                    scov_pp[alpha, beta] += (E0[B]/G_B) * int_pp_E
                    serr_pp[alpha, beta] = np.sqrt(serr_pp[alpha, beta]**2 + ((E0[B]/G_B) * err_pp_E)**2)

                    ncov_px[alpha, beta] += (sigma_E**2/G_B) * int_px_E
                    nerr_px[alpha, beta] = np.sqrt(nerr_px[alpha, beta]**2 + ((sigma_E**2/G_B) * err_px_E)**2)
                    scov_px[alpha, beta] += (E0[B]/G_B) * int_px_E
                    serr_px[alpha, beta] = np.sqrt(serr_px[alpha, beta]**2 + ((E0[B]/G_B) * err_px_E)**2)

                    ncov_xp[alpha, beta] += (sigma_E**2/G_B) * int_xp_E
                    nerr_xp[alpha, beta] = np.sqrt(nerr_xp[alpha, beta]**2 + ((sigma_E**2/G_B) * err_xp_E)**2)
                    scov_xp[alpha, beta] += (E0[B]/G_B) * int_xp_E
                    serr_xp[alpha, beta] = np.sqrt(serr_xp[alpha, beta]**2 + ((E0[B]/G_B) * err_xp_E)**2)

                    ncov_xx[alpha, beta] += (sigma_E**2/G_B) * int_xx_E
                    nerr_xx[alpha, beta] = np.sqrt(nerr_xx[alpha, beta]**2 + ((sigma_E**2/G_B) * err_xx_E)**2)
                    scov_xx[alpha, beta] += (E0[B]/G_B) * int_xx_E
                    serr_xx[alpha, beta] = np.sqrt(serr_xx[alpha, beta]**2 + ((E0[B]/G_B) * err_xx_E)**2)
                
                    Omega_anb = annuli_intersection_area(rs1[alpha], rs1[alpha+1], rs2[beta], rs2[beta+1])
    
                    if alpha == beta:
                        
                        cterm_n = ( (1/4) * (1/Nlens) * (1/G_B) 
                                  * ( sigma_L**2 * sigma_E**2
                                    + sigma_L**2 * E0[B]
                                    + L0 * sigma_E**2 ) 
                                   * Omegatot / Omegas1[alpha] )
                        
                        cterm_s = ( (1/4) * (1/Nlens) * (1/G_B) 
                                  * L0 * E0[B] 
                                   * Omegatot / Omegas1[alpha])
                    
                        ncov_pp[alpha, beta] += cterm_n
                        ncov_xx[alpha, beta] += cterm_n
                        
                        scov_pp[alpha, beta] += cterm_s
                        scov_xx[alpha, beta] += cterm_s
    
            test_err(nerr_pp[alpha, beta], ncov_pp[alpha, beta], f'LELE ncov plus plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(nerr_px[alpha, beta], ncov_px[alpha, beta], f'LELE ncov plus times redshift bins {B,D} angular bins {alpha, beta}')
            test_err(nerr_xp[alpha, beta], ncov_xp[alpha, beta], f'LELE ncov times plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(nerr_xx[alpha, beta], ncov_xx[alpha, beta], f'LELE ncov times times redshift bins {B,D} angular bins {alpha, beta}')
            
            test_err(serr_pp[alpha, beta], scov_pp[alpha, beta], f'LELE scov plus plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(serr_px[alpha, beta], scov_px[alpha, beta], f'LELE scov plus times redshift bins {B,D} angular bins {alpha, beta}')
            test_err(serr_xp[alpha, beta], scov_xp[alpha, beta], f'LELE scov times plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(serr_xx[alpha, beta], scov_xx[alpha, beta], f'LELE scov times times redshift bins {B,D} angular bins {alpha, beta}')

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

