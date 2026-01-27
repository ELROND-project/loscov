import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from redshift_distributions import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('LLp','LLx', 'LEp', 'LEx', 'LP', 'EP', 'angular_distributions', 'redshift_distributions', 'L0', 'E0')

################################################## LELP cosmic covariance ##############################################################

def generate_ccov_LELP(B, D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy shape cross LOS shear - galaxy position correlation functions.
    
    B             : the galaxy redshift bin D (0 to Nbinz_E)
    D             : the galaxy redshift bin D (0 to Nbinz_P)
    """

    def generate_matrices(sign):    
    
        angular_distribution1 = angular_distributions[f'LE_{sign}'][B]
        angular_distribution2 = angular_distributions['LP'][D]
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LE 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LE (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LP 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LP (in rad)
        
        # Initialise the blocks
        ccov_p = np.zeros((Nbin1, Nbin2))
        ccov_x = np.zeros((Nbin1, Nbin2))
        
        err_p = np.zeros((Nbin1, Nbin2))
        err_x = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands
        
        def integrand_p(params):
            
            psi_b, psi_kd, r_b, r_kd, r_k = params
        
            x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
            y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
            
            r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
    
            f = ( EP[B][D](r_bd) * cos2(psi_bd - psi_b)
              * ( LLp(r_k) * cos2(psi_b) * cos2(psi_kd) 
                + LLx(r_k) * sin2(psi_b) * sin2(psi_kd) )
                + LP[D](r_bd) * cos2(psi_bd-psi_b)
              * ( LEp[B](r_k) * cos2(psi_b) * cos2(psi_kd) 
                + LEx[B](r_k) * sin2(psi_b) * sin2(psi_kd))
                )
            
            f *= 2 * np.pi * r_k * r_b * r_kd
            
            return f
        
        def integrand_x(params):
            
            psi_b, psi_kd, r_b, r_kd, r_k = params
        
            x_bd = r_kd * np.cos(psi_kd) - r_b * np.cos(psi_b) + r_k
            y_bd = r_kd * np.sin(psi_kd) - r_b * np.sin(psi_b)
            
            r_bd = np.sqrt( x_bd**2 + y_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
    
            f = ( EP[B][D](r_bd) * cos2(psi_bd - psi_b)
              * ( LLx(r_k) * cos2(psi_b) * sin2(psi_kd) 
                - LLp(r_k) * sin2(psi_b) * cos2(psi_kd) )
                + LP[D](r_bd) * cos2(psi_bd-psi_b)
              * ( LEx[B](r_k) * cos2(psi_b) * sin2(psi_kd) 
                - LEp[B](r_k) * sin2(psi_b) * cos2(psi_kd))
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
                         
                ccov_p[alpha, beta], err_p[alpha, beta] = integral_bins(integrand_p, alpha, beta)
                ccov_x[alpha, beta], err_x[alpha, beta] = integral_bins(integrand_x, alpha, beta)
    
                test_err(err_p[alpha, beta], ccov_p[alpha, beta], f'LELP ccov plus redshifts {B, D} angular bins {alpha, beta}')
                test_err(err_x[alpha, beta], ccov_x[alpha, beta], f'LELP ccov times redshifts {B, D} angular bins {alpha, beta}')

        err = np.sqrt(err_p**2+err_x**2)
        
        return ccov_p, ccov_x, err

    #plus

    ccov_p, ccov_x, errp = generate_matrices('plus')
    
    ccovp = ccov_p + ccov_x
    
    #minus

    ccov_p, ccov_x, errm = generate_matrices('minus')
    
    ccovm = ccov_p - ccov_x
    
    ccov = np.block([[ccovm],
                     [ccovp]])
    
    err = np.block([[errm],
                     [errp]])
    
    return ccov, err
    
################################################## LELP noise/sparsity covariance #############################################################

def generate_ncov_LELP(B, D):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy shape cross LOS shear - galaxy position correlation functions.
    
    B             : the galaxy redshift bin D (0 to Nbinz_E)
    D             : the galaxy redshift bin D (0 to Nbinz_P)
    """

    def generate_matrices(sign):    

        angular_distribution1 = angular_distributions[f'LE_{sign}'][B]
        angular_distribution2 = angular_distributions['LP'][D]
        
        Nbin1       = angular_distribution1.Nbina      #the number of angular bins for LE 
        Omegas1     = angular_distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
        rs1         = angular_distribution1.limits     #the angular bin limits for LE (in rad)
        
        Nbin2       = angular_distribution2.Nbina      #the number of angular bins for LP 
        Omegas2     = angular_distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
        rs2         = angular_distribution2.limits     #the angular bin limits for LP (in rad)
        
        # Initialise the blocks
        ncov_p = np.zeros((Nbin1, Nbin2))
        ncov_x = np.zeros((Nbin1, Nbin2))
        
        nerr_p = np.zeros((Nbin1, Nbin2))
        nerr_x = np.zeros((Nbin1, Nbin2))
        
        scov_p = np.zeros((Nbin1, Nbin2))
        scov_x = np.zeros((Nbin1, Nbin2))
        
        serr_p = np.zeros((Nbin1, Nbin2))
        serr_x = np.zeros((Nbin1, Nbin2))
        
        # Define the integrands
        
        def integrand_p(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
    
            f = EP[B][D](r_bd) * cos2(psi_d) * cos2(psi_bd)
    
            f *= np.pi * r_b * r_d        #factor of 2 cancels with half out the front
                                  
            return f
        
        def integrand_x(params):
            
            r_b, r_d, psi_d = params
        
            y_bd = r_d*np.sin(psi_d)
            x_bd = r_d*np.cos(psi_d) - r_b
            
            r_bd = np.sqrt( y_bd**2 + x_bd**2 ) 
            psi_bd = np.arctan2(y_bd, x_bd)
    
            f = EP[B][D](r_bd) * sin2(psi_d) * sin2(psi_bd)
    
            f *= np.pi * r_b * r_d        #factor of 2 cancels with half out the front
                                  
            return f
        
        def integral_bins(integrand, alpha, beta):
            
            ranges = [(rs1[alpha], rs1[alpha+1]), (rs2[beta], rs2[beta+1]), (0, 2*np.pi)]
            
            integral, err = monte_carlo_integrate(integrand, ranges, Nsamp)
            
            # normalisation of differential elements
            integral /= Omegas1[alpha] * Omegas2[beta]
            err      /= Omegas1[alpha] * Omegas2[beta]
            
            return integral, err
        
        for alpha in range(Nbin1):
            for beta in range(Nbin2): 
    
                int_p, err_p = integral_bins(integrand_p, alpha, beta)
                int_x, err_x = integral_bins(integrand_x, alpha, beta)
                         
                ncov_p[alpha, beta] = (sigma_L**2/Nlens) * int_p
                nerr_p[alpha, beta] = (sigma_L**2/Nlens) * err_p
                         
                scov_p[alpha, beta] = (L0/Nlens) * int_p
                serr_p[alpha, beta] = (L0/Nlens) * err_p
                
                ncov_x[alpha, beta] = (sigma_L**2/Nlens) * int_x
                nerr_x[alpha, beta] = (sigma_L**2/Nlens) * err_x
                         
                scov_x[alpha, beta] = (L0/Nlens) * int_x
                serr_x[alpha, beta] = (L0/Nlens) * err_x
    
                test_err(nerr_p[alpha, beta], ncov_p[alpha, beta], f'LELP ncov plus redshifts {B, D} angular bins {alpha, beta}')
                test_err(nerr_x[alpha, beta], ncov_x[alpha, beta], f'LELP ncov times redshifts {B, D} angular bins {alpha, beta}')
    
                test_err(serr_p[alpha, beta], scov_p[alpha, beta], f'LELP scov plus redshifts {B, D} angular bins {alpha, beta}')
                test_err(serr_x[alpha, beta], scov_x[alpha, beta], f'LELP scov times redshifts {B, D} angular bins {alpha, beta}')

        nerr = np.sqrt(nerr_p**2+nerr_x**2)
        serr = np.sqrt(serr_p**2+serr_x**2)  
        
        return ncov_p, ncov_x, scov_p, scov_x, nerr, serr

    #plus

    ncov_p, ncov_x, scov_p, scov_x, nerrp, serrp = generate_matrices('plus')
    
    ncovp = ncov_p + ncov_x
    scovp = scov_p + scov_x

    #minus

    ncov_p, ncov_x, scov_p, scov_x, nerrm, serrm = generate_matrices('minus')
    
    ncovm = ncov_p - ncov_x
    scovm = scov_p - scov_x
    
    ncov = np.block([[ncovm],
                     [ncovp]])
    
    nerr = np.block([[nerrm],
                     [nerrp]])
    
    scov = np.block([[scovm],
                     [scovp]])
    
    serr = np.block([[serrm],
                     [serrp]])
    
    return [ncov, scov], [nerr, serr]
