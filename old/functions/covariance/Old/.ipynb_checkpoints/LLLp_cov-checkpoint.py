import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('xi2_LOS_plus_intp','xi1_LOS_plus_intp','xi32_LOS_plus_intp','xi2_LOS_minus_intp','xi1_LOS_minus_intp','xi32_LOS_minus_intp')
get_item('xi2_d_intp', 'xi1_d_intp')
get_item('xi2_LOS_d_intp', 'xi1_LOS_d_intp', 'xi32_LOS_d2_intp', 'xi32_LOS2_d_intp')

############################################################ 6.5 LLLp ########################################################################
##############################################################################################################################################

################################################# 6.5.1 LLLp noise covariance ################################################################

def generate_ncov_LLLp(distributions, sigma_noise, sigma_shape, b1, b2, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear - LOS shear x LOS shear - galaxy position 
    correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Lp']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina      #the number of angular bins for Lp 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Lp (in rad)
        
    nsamples_p = load_file("nsample_dicts/LLLp/ncovp") if use_measured_samples and os.path.exists("nsample_dicts/LLLp/ncovp") else None
    nsamples_m = load_file("nsample_dicts/LLLp/ncovm") if use_measured_samples and os.path.exists("nsample_dicts/LLLp/ncovm") else None
		
    def I_s3(a1, a2, eta, nsampl):
        """
        This function computes the third term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
        """
		
        def integrand_s3(params):

            phi_l, l, phi_g, g = params
            Theta_gl = np.sqrt( g**2 + l**2 - 2*g*l*np.cos(phi_g-phi_l) )
            f = np.cos(2*(phi_g-phi_l+eta*phi_l)) * xi2_LOS_d_intp[b1](Theta_gl) *g*l
            
            return f
		
        integral, err = monte_carlo_integrate(integrand_s3, [(0, 2*np.pi), (rs2[a2], rs2[a2+1]), (0, 2*np.pi), (rs1[a1], rs1[a1+1])], nsampl)

        integral /= ( Omegas1[a1]*Omegas2[a2] )
        err /= ( Omegas1[a1]*Omegas2[a2] ) 
        
        return integral, err

    ncov_p = np.zeros((Nbin1,Nbin2))
    ncov_m = np.zeros((Nbin1,Nbin2))
        
        # Compute and add the integral contribution
    
    for a1 in range(Nbin1):

        for a2 in range(Nbin2): # this could be made more efficient if the matrix is symmetric (dunno)
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_p = int(nsamples_p[str(b1)][str(a1) + str(a2)]) if nsamples_p else nsamp
            nsample_m = int(nsamples_p[str(b1)][str(a1) + str(a2)]) if nsamples_p else nsamp

            Is3p, errp = I_s3(a1, a2, 1, nsample_p)
            Is3m, errm = I_s3(a1, a2, -1, nsample_m)
            
            ncov_p[a1, a2] = -Is3p*sigma_noise**2/Nlens
            ncov_m[a1, a2] = -Is3m*sigma_noise**2/Nlens
            
            errp = errp*sigma_noise**2/Nlens
            errm = errm*sigma_noise**2/Nlens 

            test_err(errp, ncov_p[a1, a2], f'LLLp ncov p redshifts {b1} angular bins {a1, a2}')
            test_err(errm, ncov_m[a1, a2], f'LLLp ncov m redshifts {b1} angular bins {a1, a2}')

    ncov = np.block([[ncov_p],
                     [ncov_m]])
    
    return ncov, ncov_p, ncov_m, None, None

################################################# 6.6.2 LLLp cosmic covariance ###############################################################

def generate_ccov_LLLp(distributions, b1, b2, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - LOS shear x LOS shear - galaxy position 
    correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b  (0 to 4) (irrelevant for this matrix, usually set to None)
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Lp']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina      #the number of angular bins for Lp 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Lp (in rad)
        
    nsamples_p = load_file("nsample_dicts/LLLp/ccovp") if use_measured_samples and os.path.exists("nsample_dicts/LLLp/ccovp") else None
    nsamples_m = load_file("nsample_dicts/LLLp/ccovm") if use_measured_samples and os.path.exists("nsample_dicts/LLLp/ccovm") else None

    def I_c(a1,a2,eta):
        
        def func_c(params):
            # unpack the parameters
            k, g, phi_g, h, psi_kh = params 
            # we need to compute additional angles that are not directly integration variables
            Theta_kg = np.sqrt( g**2 + k**2 - 2*g*k*np.cos(phi_g) )
            Theta_h = np.sqrt( h**2 + k**2 + 2*h*k*np.cos(psi_kh) )
            arg = (k+h*np.cos(psi_kh))/Theta_h
            # because of numeric instability I must make sure the result is between -1 and 1, as I will apply the arccos
            arg = np.clip(arg, -1, 1)
            # use of arccos
            phi_h = np.arccos(arg)
            phi_h[psi_kh > np.pi] *= -1 # because of how arccos works; it returns a value between 0 and pi, I must correct for that
            
            # final calculation
            f = k*g*h*( xi2_LOS_plus_intp(Theta_h)*np.cos(2*(phi_g - psi_kh + eta*psi_kh)) 
                + xi2_LOS_minus_intp(Theta_h)*np.cos(2*(2*phi_h-phi_g - psi_kh + eta*psi_kh)) )*xi2_LOS_d_intp[b1](Theta_kg)
            prefactor = -2*np.pi/( Omegatot*Omegas1[a1]*Omegas2[a2] )
        
            return f*prefactor
        
        # Use preloaded nsamples or fall back to nsamp
        if eta == 1:
            nsample = int(nsamples_p[str(b1)][str(a1) + str(a2)]) if nsamples_p else nsamp
        else:
            nsample = int(nsamples_m[str(b1)][str(a1) + str(a2)]) if nsamples_m else nsamp

        integral, err = monte_carlo_integrate(func_c, [(0, r2_max), (rs1[a1], rs1[a1+1]), (0, 2*np.pi), (rs2[a2], rs2[a2+1]), (0, 2*np.pi)], nsample) 

        return integral, err 
        
    ccov_p = np.zeros((Nbin1,Nbin2))
    ccov_m = np.zeros((Nbin1,Nbin2))

    # The number of samples to reach the 10% precision here is much smaller (5e5), it runs in 0.15s. It means we can go to a far greater precision here.
    # Anyway the nsamp I leave here is again aimed to achieve the 10% precison, just like for the terms above
    
    for a1 in range(Nbin1):

        for a2 in range(Nbin2): # this could be made more efficient if the matrix is symmetric (dunno)

            ccov_p[a1, a2], errp = I_c(a1,a2,1)
            ccov_m[a1, a2], errm = I_c(a1,a2,-1)

            test_err(errp, ccov_p[a1, a2], f'LLLp ccov p redshifts {b1} angular bins {a1, a2}')
            test_err(errm, ccov_m[a1, a2], f'LLLp ccov m redshifts {b1} angular bins {a1, a2}')

    ccov = np.block([[ccov_p],
                     [ccov_m]])
    
    
    return ccov, ccov_p, ccov_m, None, None


################################################# 6.6.3 LLLp sparsity covariance #############################################################

def generate_scov_LLLp(distributions, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the LOS shear - LOS shear x LOS shear - galaxy position 
    correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b  (0 to 4) 
    b2             : the galaxy redshift bin b' (0 to 4) (irrelevant for this matrix, usually set to None)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['LL']
    distribution2 = distributions['Lp']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for LL 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for LL (in rad)
    
    Nbin2       = distribution2.Nbina      #the number of angular bins for Lp 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Lp (in rad)
	
    def I_s1(a1, a2, eta):
        """
        This function computes the first term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
        """

        def integrand_s11(l):
            return xi32_LOS_d2_intp[b1](l)*2*l

        def integrand_s12(l):
            if eta > 0:
                return xi32_LOS_plus_intp(l)*2*l
            else:
                return xi32_LOS_minus_intp(l)*2*l
            
        integral1, err1 = monte_carlo_integrate(integrand_s11, [(rs1[a1], rs1[a1+1])], 7e2)
        integral2, err2 = monte_carlo_integrate(integrand_s12, [(rs2[a2], rs2[a2+1])], 1e2)

        result = 2*(np.pi**2)*integral1*integral2 /( Omegas1[a1]*Omegas2[a2] )
        err = np.sqrt( (2*(np.pi**2)*err1*integral2 /( Omegas1[a1]*Omegas2[a2]))**2 + (2*(np.pi**2)*integral1*err2 /( Omegas1[a1]*Omegas2[a2] ))**2 ) 
        
        return result, err
		
	
    def I_s2(a1, a2, eta):
        """
        This function computes the second term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
        """

        def integrand_s21(l):
            return xi2_LOS_d_intp[b1](l)*2*l

        def integrand_s22(l):
            if eta > 0:
                return xi2_LOS_plus_intp(l)*2*l
            else:
                return xi2_LOS_minus_intp(l)*2*l

        integral1, err1 = monte_carlo_integrate(integrand_s21, [(rs1[a1], rs1[a1+1])], 7e2)
        integral2, err2 = monte_carlo_integrate(integrand_s22, [(rs2[a2], rs2[a2+1])], 1e2)

        result = -2*(np.pi**2)*integral1*integral2 /( Omegas1[a1]*Omegas2[a2] )
        err = np.sqrt( (2*(np.pi**2)*err1*integral2 /( Omegas1[a1]*Omegas2[a2]))**2 + (2*(np.pi**2)*integral1*err2 /( Omegas1[a1]*Omegas2[a2] ))**2 ) 

        return result, err
        
    nsamples_p = load_file("nsample_dicts/LLLp/scovp") if use_measured_samples and os.path.exists("nsample_dicts/LLLp/scovp") else None
    nsamples_m = load_file("nsample_dicts/LLLp/scovm") if use_measured_samples and os.path.exists("nsample_dicts/LLLp/scovm") else None
		
    def I_s3(a1, a2, eta):
        """
        This function computes the third term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
        """
		
        def integrand_s3(params):

            phi_l, l, phi_g, g = params
		    
            Theta_gl = np.sqrt( g**2 + l**2 - 2*g*l*np.cos(phi_g-phi_l) )

            f = np.cos(2*(phi_g-phi_l+eta*phi_l)) * xi2_LOS_d_intp[b1](Theta_gl) *g*l

            return f
        
        # Use preloaded nsamples or fall back to nsamp
        if eta == 1:
            nsample = int(nsamples_p[str(b1)][str(a1) + str(a2)]) if nsamples_p else nsamp
        else:
            nsample = int(nsamples_m[str(b1)][str(a1) + str(a2)]) if nsamples_m else nsamp
        
        integral, err = monte_carlo_integrate(integrand_s3, [(0, 2*np.pi), (rs2[a2], rs2[a2+1]), (0, 2*np.pi), (rs1[a1], rs1[a1+1])], nsample) 

        result = integral * xi1_LOS_plus_intp(0) /( (Omegas1[a1])*(Omegas2[a2]) )
        err = err * xi1_LOS_plus_intp(0) /( (Omegas1[a1])*(Omegas2[a2]) )
		
        return result, err

    scov_p = np.zeros((Nbin1,Nbin2))
    scov_m = np.zeros((Nbin1,Nbin2)) 

    for a1 in range(Nbin1):

        for a2 in range(Nbin2): # this could be made more efficient if the matrix is symmetric (dunno)
		
            Is1, err1 = I_s1(a1, a2, 1)
            Is2, err2 = I_s2(a1, a2, 1)
            Is3, err3 = I_s3(a1, a2, 1)

            scov_p[a1, a2] = -(Is1 + Is2 + Is3)/Nlens
            err_p = np.sqrt(err1**2 + err2**2 + err3**2) / Nlens
	
            Is1, err1 = I_s1(a1, a2, -1)
            Is2, err2 = I_s2(a1, a2, -1)
            Is3, err3 = I_s3(a1, a2, -1)

            scov_m[a1, a2] = -(Is1 + Is2 + Is3)/Nlens
            err_m = np.sqrt(err1**2 + err2**2 + err3**2) / Nlens

            test_err(err_p, scov_p[a1, a2], f'LLLp scov p redshifts {b1} angular bins {a1, a2}')
            test_err(err_m, scov_m[a1, a2], f'LLLp scov m redshifts {b1} angular bins {a1, a2}')
            
    scov = np.block([[scov_p],
                     [scov_m]])

    return scov, scov_p, scov_m, None, None