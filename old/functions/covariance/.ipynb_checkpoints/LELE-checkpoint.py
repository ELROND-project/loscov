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
    
        # Define the integrands
        
        def integrand_pp(params):
            
            psi_b, psi_kd, r_b, r_kd, r_k = params
        
            y_kb = r_b * np.sin(psi_b)
            x_kb = r_b * np.cos(psi_b) - r_k
            
            r_kb = np.sqrt( y_kb**2 + x_kb**2 ) 
            psi_kb = np.arctan2(y_kb, x_kb)
            
            r_bd = cos_law_side(r_kd, r_kb, (psi_kd-psi_kb))
            psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd
    
            f = ( ( LLp(r_k) * cos2(psi_b) * cos2(psi_kd)
                  + LLx(r_k) * sin2(psi_b) * sin2(psi_kd) )
            * ( EEp[B][D](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
               + EEx[B][D](r_bd) * sin2(psi_bd - psi_bd) * sin2(psi_bd - psi_kd) )
            + ( LEp[D](r_k) * cos2(psi_b) * cos2(psi_kd)
               + LEx[D](r_k) * sin2(psi_b) * sin2(psi_kd) )
            * ( LEp[B](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
               + LEx[B](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_b * r_kd
            
            return f
        
        def integrand_px(params):
            
            psi_b, psi_kd, r_b, r_kd, r_k = params
        
            y_kb = r_b * np.sin(psi_b)
            x_kb = r_b * np.cos(psi_b) - r_k
            
            r_kb = np.sqrt( y_kb**2 + x_kb**2 ) 
            psi_kb = np.arctan2(y_kb, x_kb)
            
            r_bd = cos_law_side(r_kd, r_kb, (psi_kd-psi_kb))
            psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd
    
            f = - ( ( LLp(r_k) * cos2(psi_b) * sin2(psi_kd)
                  - LLx(r_k) * sin2(psi_b) * cos2(psi_kd) )
            * ( EEp[B][D](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
               - EEx[B][D](r_bd) * sin2(psi_bd - psi_bd) * cos2(psi_bd - psi_kd) )
            + ( LEp[D](r_k) * cos2(psi_b) * sin2(psi_kd)
               - LEx[D](r_k) * sin2(psi_b) * cos2(psi_kd) )
                * ( LEp[B](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
                  - LEx[B](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_b * r_kd
            
            return f
        
        def integrand_xp(params):
            
            psi_b, psi_kd, r_b, r_kd, r_k = params
        
            y_kb = r_b * np.sin(psi_b)
            x_kb = r_b * np.cos(psi_b) - r_k
            
            r_kb = np.sqrt( y_kb**2 + x_kb**2 ) 
            psi_kb = np.arctan2(y_kb, x_kb)
            
            r_bd = cos_law_side(r_kd, r_kb, (psi_kd-psi_kb))
            psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd
    
            f = - ( ( LLp(r_k) * sin2(psi_b) * cos2(psi_kd)
                  - LLx(r_k) * cos2(psi_b) * sin2(psi_kd) )
            * ( EEp[B][D](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
               - EEx[B][D](r_bd) * cos2(psi_bd - psi_bd) * sin2(psi_bd - psi_kd) )
            + ( LEp[D](r_k) * sin2(psi_b) * cos2(psi_kd)
               - LEx[D](r_k) * cos2(psi_b) * sin2(psi_kd) )
                * ( LEp[B](r_bd) * sin2(psi_bd - psi_b) * cos2(psi_bd - psi_kd)
                  - LEx[B](r_bd) * cos2(psi_bd - psi_b) * sin2(psi_bd - psi_kd) )
                )
            
            f *= 2 * np.pi * r_k * r_b * r_kd
            
            return f
        
        def integrand_xx(params):
            
            psi_b, psi_kd, r_b, r_kd, r_k = params
        
            y_kb = r_b * np.sin(psi_b)
            x_kb = r_b * np.cos(psi_b) - r_k
            
            r_kb = np.sqrt( y_kb**2 + x_kb**2 ) 
            psi_kb = np.arctan2(y_kb, x_kb)
            
            r_bd = cos_law_side(r_kd, r_kb, (psi_kd-psi_kb))
            psi_bd = cos_law_angle(r_kd, r_bd, r_kb) + psi_kd
    
            f = ( ( LLp(r_k) * sin2(psi_b) * sin2(psi_kd)
                  + LLx(r_k) * cos2(psi_b) * cos2(psi_kd) )
                * ( EEp[B][D](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
                 + EEx[B][D](r_bd) * cos2(psi_bd - psi_bd) * cos2(psi_bd - psi_kd) )
                + ( LEp[D](r_k) * sin2(psi_b) * sin2(psi_kd)
                  + LEx[D](r_k) * cos2(psi_b) * cos2(psi_kd) )
                * ( LEp[B](r_bd) * sin2(psi_bd - psi_b) * sin2(psi_bd - psi_kd)
                  + LEx[B](r_bd) * cos2(psi_bd - psi_b) * cos2(psi_bd - psi_kd) )
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
                         
                ccov_pp[alpha, beta], err_pp = integral_bins(integrand_pp, alpha, beta)
                ccov_px[alpha, beta], err_px = integral_bins(integrand_px, alpha, beta)
                ccov_xp[alpha, beta], err_xp = integral_bins(integrand_xp, alpha, beta)
                ccov_xx[alpha, beta], err_xx = integral_bins(integrand_xx, alpha, beta)
        
                test_err(err_pp, ccov_pp[alpha, beta], f'LELE ccov plus plus redshift bins{B, D} angular bins {alpha, beta}')
                test_err(err_px, ccov_px[alpha, beta], f'LELE ccov plus times redshift bins{B, D} angular bins {alpha, beta}')
                test_err(err_xp, ccov_xp[alpha, beta], f'LELE ccov times plus redshift bins{B, D} angular bins {alpha, beta}')
                test_err(err_xx, ccov_xx[alpha, beta], f'LELE ccov times times redshift bins{B, D} angular bins {alpha, beta}')

        return ccov_pp, ccov_px, ccov_xp, ccov_xx

    #ccov_pp

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('plus', 'plus')
    
    ccovpp = ccov_pp + ccov_px + ccov_xp + ccov_xx

    #ccov_pm

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('plus', 'minus')
    
    ccovpm = ccov_pp - ccov_px + ccov_xp - ccov_xx

    #ccov_mp

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('minus', 'plus')

    ccovmp = ccov_pp + ccov_px - ccov_xp - ccov_xx

    #ccov_mm

    ccov_pp, ccov_px, ccov_xp, ccov_xx = generate_matrices('minus', 'minus')
    
    ccovmm = ccov_pp - ccov_px - ccov_xp + ccov_xx

    
    ccov = np.block([[ccovmm, ccovmp],
                     [ccovpm, ccovpp]])
    
    return ccov

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
        
        scov_pp = np.zeros((Nbin1, Nbin2))
        scov_px = np.zeros((Nbin1, Nbin2))
        scov_xp = np.zeros((Nbin1, Nbin2))
        scov_xx = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands
        
        def integrand_pp_L(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = ( cos2(psi_d) 
                * ( EEp[B][D](r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d) 
                  + EEx[B][D](r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d) 
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_pp_E(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = ( cos2(psi_d) 
                * ( LLp(r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d)  
                  + LLx(r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d)  
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_px_L(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = - ( sin2(psi_d) 
                * ( EEp[B][D](r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d)  
                  - EEx[B][D](r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d)  
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_px_E(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = - ( sin2(psi_d) 
                * ( LLp(r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d) 
                  - LLx(r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d) 
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_xp_L(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = ( sin2(psi_d) 
                * ( EEp[B][D](r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d)  
                  - EEx[B][D](r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d)  
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_xp_E(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = ( sin2(psi_d) 
                * ( LLp(r_bd) * sin2(psi_bd) * cos2(psi_bd - psi_d)   
                  - LLx(r_bd) * cos2(psi_bd) * sin2(psi_bd - psi_d) 
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_xx_L(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = ( cos2(psi_d) 
                * ( EEp[B][D](r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d)  
                  + EEx[B][D](r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d)  
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        def integrand_xx_E(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
            
            f = ( cos2(psi_d) 
                * ( LLp(r_bd) * sin2(psi_bd) * sin2(psi_bd - psi_d) 
                  + LLx(r_bd) * cos2(psi_bd) * cos2(psi_bd - psi_d) 
                ) )
    
            f *= np.pi * r_b * r_d #the usual 2 disappears because of the factor 1/2 out the front
                                  
            return f
        
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
            
            # normalisation of differential elements
            integral /= (Omegas1[alpha] * Omegas2[beta]) 
            err      /= (Omegas1[alpha] * Omegas2[beta]) 
            
            return integral, err
        
        integrand_pp = [integrand_pp_L]
        integrand_px = [integrand_px_L]
        integrand_xp = [integrand_xp_L]
        integrand_xx = [integrand_xx_L]
    
        if B == D:
            integrand_pp.append(integrand_pp_E)
            integrand_px.append(integrand_px_E)
            integrand_xp.append(integrand_xp_E)
            integrand_xx.append(integrand_xx_E)
    
        for alpha in range(Nbin1):
    
            for beta in range(Nbin2): 
                         
                int_pp, err_pp = integral_bins(integrand_pp, alpha, beta)
                int_px, err_px = integral_bins(integrand_px, alpha, beta)
                int_xp, err_xp = integral_bins(integrand_xp, alpha, beta)
                int_xx, err_xx = integral_bins(integrand_xx, alpha, beta)
    
                ncov_pp[alpha, beta] = (sigma_L**2/Nlens) * int_pp[0]
    
                nerr_pp = (sigma_L**2/Nlens) * err_pp[0]
    
                scov_pp[alpha, beta] = (L0/Nlens) * int_pp[0]
    
                serr_pp = (L0/Nlens) * err_pp[0]
    
                ncov_px[alpha, beta] = (sigma_L**2/Nlens) * int_px[0]
    
                nerr_px = (sigma_L**2/Nlens) * err_px[0]
    
                scov_px[alpha, beta] = (L0/Nlens) * int_px[0]
    
                serr_px = (L0/Nlens) * err_px[0]
    
                ncov_xp[alpha, beta] = (sigma_L**2/Nlens) * int_xp[0]
    
                nerr_xp = (sigma_L**2/Nlens) * err_xp[0]
    
                scov_xp[alpha, beta] = (L0/Nlens) * int_xp[0]
    
                serr_xp = (L0/Nlens) * err_xp[0]
    
                ncov_xx[alpha, beta] = (sigma_L**2/Nlens) * int_xx[0]
    
                nerr_xx = (sigma_L**2/Nlens) * err_xx[0]
    
                scov_xx[alpha, beta] = (L0/Nlens) * int_xx[0]
    
                serr_xx = (L0/Nlens) * err_xx[0]
            
                #addition of constant term and term with sigma_E
                if B == D:
        
                    ncov_pp[alpha, beta] += (sigma_E**2/G_B) * int_pp[1] 
        
                    nerr_pp = (
                             np.sqrt( nerr_pp**2
                                + ( (sigma_E**2/G_B) * err_pp[1])**2 )
                              ) 
        
                    scov_pp[alpha, beta] += (E0[B]/G_B) * int_pp[1] 
        
                    serr_pp = (
                              np.sqrt(  serr_pp**2
                                      + ( (E0[B]/G_B) * err_pp[1])**2 )
                              )
        
                    ncov_px[alpha, beta] += (sigma_E**2/G_B) * int_px[1] 
        
                    nerr_px = (
                              np.sqrt(  nerr_px**2
                                      + ( (sigma_E**2/G_B) * err_px[1])**2 )
                              )
        
                    scov_px[alpha, beta] += (E0[B]/G_B) * int_px[1] 
        
                    serr_px = (
                              np.sqrt(  serr_px**2
                                      + ( (E0[B]/G_B) * err_px[1])**2 )
                              )
        
                    ncov_xp[alpha, beta] += (sigma_E**2/G_B) * int_xp[1] 
        
                    nerr_xp = (
                              np.sqrt(  nerr_xp**2
                                      + ( (sigma_E**2/G_B) * err_xp[1])**2 )
                              )
        
                    scov_xp[alpha, beta] += (E0[B]/G_B) * int_xp[1] 
        
                    serr_xp = (
                              np.sqrt(  serr_xp**2
                                      + ( (E0[B]/G_B) * err_xp[1])**2 )
                              )
        
                    ncov_xx[alpha, beta] += (sigma_E**2/G_B) * int_xx[1] 
        
                    nerr_xx = (
                              np.sqrt(  nerr_xx**2
                                      + ( (sigma_E**2/G_B) * err_xx[1])**2 )
                              )
        
                    scov_xx[alpha, beta] += (E0[B]/G_B) * int_xx[1] 
        
                    serr_xx = (
                              np.sqrt(  serr_xx**2
                                      + ( (E0[B]/G_B) * err_xx[1])**2 )
                              )
                
                    Omega_anb = annuli_intersection_area(rs1[alpha], rs1[alpha+1], rs2[beta], rs2[beta+1])
    
                    if Omega_anb != 0:
                        
                        cterm_n = ( (1/4) * (1/Nlens) * (1/G_B) 
                                  * ( sigma_L**2 * sigma_E**2
                                    + sigma_L**2 * E0[B]
                                    + L0 * sigma_E**2 )
                                   * Omega_anb * Omegatot / ( Omegas1[alpha] * Omegas2[beta] ) )
                        
                        cterm_s = ( (1/4) * (1/Nlens) * (1/G_B) 
                                  * L0 * E0[B]
                                   * Omega_anb * Omegatot / ( Omegas1[alpha] * Omegas2[beta] ))
                    
                        ncov_pp[alpha, beta] += cterm_n
                        ncov_xx[alpha, beta] += cterm_n
                        
                        scov_pp[alpha, beta] += cterm_s
                        scov_xx[alpha, beta] += cterm_s
    
            test_err(nerr_pp, ncov_pp[alpha, beta], f'LELE ncov plus plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(nerr_px, ncov_px[alpha, beta], f'LELE ncov plus times redshift bins {B,D} angular bins {alpha, beta}')
            test_err(nerr_xp, ncov_xp[alpha, beta], f'LELE ncov times plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(nerr_xx, ncov_xx[alpha, beta], f'LELE ncov times times redshift bins {B,D} angular bins {alpha, beta}')
            
            test_err(serr_pp, scov_pp[alpha, beta], f'LELE scov plus plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(serr_px, scov_px[alpha, beta], f'LELE scov plus times redshift bins {B,D} angular bins {alpha, beta}')
            test_err(serr_xp, scov_xp[alpha, beta], f'LELE scov times plus redshift bins {B,D} angular bins {alpha, beta}')
            test_err(serr_xx, scov_xx[alpha, beta], f'LELE scov times times redshift bins {B,D} angular bins {alpha, beta}')

        return ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx    
    
    #plus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('plus', 'plus')
    
    ncovpp = ncov_pp + ncov_px + ncov_xp + ncov_xx    
    scovpp = scov_pp + scov_px + scov_xp + scov_xx

    #plus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('plus', 'minus')
    
    ncovpm = ncov_pp - ncov_px + ncov_xp - ncov_xx
    scovpm = scov_pp - scov_px + scov_xp - scov_xx

    #minus plus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('minus', 'plus')
    
    ncovmp = ncov_pp + ncov_px - ncov_xp - ncov_xx
    scovmp = scov_pp + scov_px - scov_xp - scov_xx

    #minus minus

    ncov_pp, ncov_px, ncov_xp, ncov_xx, scov_pp, scov_px, scov_xp, scov_xx = generate_matrices('minus', 'minus')
    
    ncovmm = ncov_pp - ncov_px - ncov_xp + ncov_xx
    scovmm = scov_pp - scov_px - scov_xp + scov_xx

    
    ncov = np.block([[ncovmm, ncovmp],
                     [ncovpm, ncovpp]])
    
    scov = np.block([[scovmm, scovmp],
                     [scovpm, scovpp]])
    
    return [ncov, scov]

