import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('xi2_LOS_plus_intp','xi1_LOS_plus_intp','xi32_LOS_plus_intp','xi2_LOS_minus_intp','xi1_LOS_minus_intp','xi32_LOS_minus_intp')
get_item('xi2_d_intp')
get_item('xi2_LOS_d_intp', 'xi32_LOS_d2_intp')

############################################################ 6.4 LpLp ########################################################################
##############################################################################################################################################

################################################# 6.4.1 LpLp noise covariance ################################################################

def generate_ncov_LpLp(distributions, sigma_noise, sigma_shape, b1, b2, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear - galaxy position correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution = distributions['Lp']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
    
    def func_n1(params):
        # unpack the parameters
        g, phi_g, h, phi_h = params 
        # we need to compute an additional angle that is not directly an integration variable
        Theta_gh = np.sqrt( g**2 + h**2 - 2*g*h*np.cos(phi_g-phi_h) )
        # final calculation
        f = g*h*np.cos(2*(phi_h-phi_g)) * xi2_d_intp[b1][b2](Theta_gh)   
        prefactor = 1/( Omegas[a1]*Omegas[a2] )
    
        return f*prefactor

    def I_n3(a1, a2, b1, b2):
        """
        This function computes the I_n3 term of the noise covariance, for bins defined by a1 and
        a2. Those bins are expressed in arcmin. b1 and b2 are the redshift bins under consideration.
        """
		
        if a1 != a2 or b1 != b2:
            return 0
        
        else:
            return 1 /( Omegas[a1]*n_b ) 

    ncov = np.zeros((Nbin,Nbin))
        
    nsamples = load_file("nsample_dicts/LpLp/ncov") if use_measured_samples and os.path.exists("nsample_dicts/LpLp/ncov") else None
        
    # Compute and add the integral contribution
    
    for a1 in range(Nbin):

        for a2 in range(Nbin): # this could be made more efficient if the matrix is symmetric (dunno)

            if b1 == b2:
        
                # Use preloaded nsamples or fall back to nsamp
                nsample = int(nsamples[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples else nsamp
                
                In2, err1 = monte_carlo_integrate(func_n1, [(rs[a1], rs[a1+1]), (0, 2*np.pi), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample) # 0.002 s (for 10% precision)
                In3 = I_n3(a1, a2, b1, b2)
                ncov[a1, a2] = (In2+In3)*sigma_noise**2/(2*Nlens)

                err1 = err1*sigma_noise**2/(2*Nlens)
                test_err(err1, ncov[a1, a2], f'LpLp ncov redshifts {b1, b2} angular bins {a1, a2}')

	
    return ncov, None, None, None, None

################################################# 6.6.2 LpLp cosmic covariance ###############################################################

def generate_ccov_LpLp(distributions, b1, b2, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - galaxy position correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution = distributions['Lp']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)

    def func_c1(params):
        # unpack the parameters
        k, g, phi_g, h, psi_kh = params 
        # we need to compute an additional angle that is not directly an integration variable
        Theta_gh = np.sqrt( g**2 + h**2 + k**2 + 2*k*(h*np.cos(psi_kh)-g*np.cos(phi_g))- 2*g*h*np.cos(phi_g-psi_kh) )
        
        # final calculation
        f = k*g*h*( xi2_LOS_plus_intp(k)*np.cos(2*(psi_kh-phi_g)) + xi2_LOS_minus_intp(k)*np.cos(2*(psi_kh+phi_g)) )*xi2_d_intp[b1][b2](Theta_gh) 
        prefactor = np.pi/( Omegatot*Omegas[a1]*Omegas[a2]) 
        
        return f*prefactor
        
        
    def func_c2(params):
        # unpack the parameters
        k, g, phi_g, h, psi_kh = params 
        # we need to compute additional angles that are not directly integration variables
        Theta_kg = np.sqrt( k**2 + g**2 - 2*k*g*np.cos(phi_g) )
        Theta_h = np.sqrt( h**2 + k**2 + 2*h*k*np.cos(psi_kh) )
    
        # final calculation
        f = k*g*h*np.cos(2*psi_kh)*np.cos(2*phi_g)*xi2_LOS_d_intp[b1](Theta_kg)*xi2_LOS_d_intp[b2](Theta_h)
        prefactor = 2*np.pi/( Omegatot*Omegas[a1]*Omegas[a2] )
        
        return f*prefactor

    ccov = np.zeros((Nbin,Nbin))
        
    nsamples1 = load_file("nsample_dicts/LpLp/ccov1") if use_measured_samples and os.path.exists("nsample_dicts/LpLp/ccov1") else None
    nsamples2 = load_file("nsample_dicts/LpLp/ccov2") if use_measured_samples and os.path.exists("nsample_dicts/LpLp/ccov2") else None
    
    for a1 in range(Nbin):

        for a2 in range(Nbin):
        
            # Use preloaded nsamples or fall back to nsamp
            nsample2 = int(nsamples2[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples2 else nsamp

            if b1 == b2:
        
                # Use preloaded nsamples or fall back to nsamp
                nsample1 = int(nsamples1[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples1 else nsamp
                
                Ic1, err1 = monte_carlo_integrate(func_c1, [(0, r2_max), (rs[a1], rs[a1+1]), (0, 2*np.pi), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample1)  

            else:
                Ic1 = 0
                err1 = 0
                
            Ic2, err2 = monte_carlo_integrate(func_c2, [(0, r2_max), (rs[a1], rs[a1+1]), (0, 2*np.pi), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample2) 
    
            ccov[a1][a2] = Ic1 + Ic2
            
            err = np.sqrt(err1**2 + err1**2)
            test_err(err, ccov[a1, a2], f'LpLp ccov redshifts {b1, b2} angular bins {a1, a2}')
            
    
    return ccov, None, None, None, None

################################################# 6.6.3 LpLp sparsity covariance #############################################################

def generate_scov_LpLp(distributions, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the LOS shear - galaxy position correlation functions.

    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution = distributions['Lp']
    
    Omegatot   = distribution.Omegatot   #\Omega in the math - the total solid angle covered by the survey
    Nbin       = distribution.Nbina      #the number of angular bins
    Omegas     = distribution.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs         = distribution.limits     #the angular bin limits (in rad)
        	
    def func_s0(params):
        if b1 != b2:
            return 0
        else:
            # unpack the parameters
            l, phi_l, k, phi_k = params 
            # we need to compute additional angles that are not directly integration variables
            Theta_lk = np.sqrt( l**2 + k**2 - 2*k*l*np.cos(phi_l-phi_k) )
            arg = (k*np.cos(phi_l-phi_k) - l)/(k**2+l**2-2*k*l*np.cos(phi_l-phi_k))**(1/2)
            # because of numeric instability I must make sure the result is between -1 and 1, as I will apply the arccos
            arg = np.clip(arg, -1, 1)
            # use of arccos
            psi_lk = np.arccos(arg)
            psi_lk[phi_l > np.pi] *= -1 # because of how arccos works; it returns a value between 0 and pi, I must correct for that
            
            # final calculation
            f = l*k*( xi2_LOS_plus_intp(Theta_lk)*np.cos(2*(phi_k-phi_l)) + xi2_LOS_minus_intp(Theta_lk)*np.cos(2*(psi_lk-phi_k-phi_l)) )
            prefactor = 1 /( 2*Omegatot*Omegas[a1]*Omegas[a2]*n_b ) #note that the (1-1/L) bit is implemented later
            
            return f*prefactor
        
        
    def func_s1(params):
        # unpack the parameters
        g, phi_g, h, phi_h = params 
        # we need to compute an additional angle that is not directly an integration variable
        Theta_gh = np.sqrt( g**2 + h**2 - 2*g*h*np.cos(phi_g-phi_h) )
        
        # final calculation
        f = g*h*np.cos(2*(phi_h-phi_g)) * xi2_d_intp[b1][b2](Theta_gh)
        prefactor = xi1_LOS_plus_intp(0)/( 2*Omegas[a1]*Omegas[a2] )
        
        return f*prefactor    

    def func_s2(params):
        # unpack the parameters
        g, h = params

        # final calculation
        f = g*h*xi32_LOS_d2_intp[b1](g)*xi32_LOS_d2_intp[b2](h)   
        prefactor = 4 * np.pi**2 /( Omegas[a1]*Omegas[a2] )

        return f*prefactor
	    
	    
    def I_s3(a1, a2):
        """
		This function computes the third term (- sign included) of the sparsity covariance, for bins defined by a1 and a2. Those bins are expressed in arcmin.
        """

        def f(x):
            return xi2_LOS_d_intp[b1](x)*2*x
            
        def g(x):
            return xi2_LOS_d_intp[b2](x)*2*x

        integralb1, errb = integrate.quad(f, rs[a1], rs[a1+1])
        integralb2, errd = integrate.quad(g, rs[a2], rs[a2+1])

        result = - integralb1 * integralb2 * np.pi**2 / (Omegas[a1] * Omegas[a2])
        err = np.sqrt( (errb * integralb2 * np.pi**2 / (Omegas[a1] * Omegas[a2]))**2 + (integralb1 * errd * np.pi**2 / (Omegas[a1] * Omegas[a2]))**2 )

        return result, err
		
		
    def I_s4(a1, a2):
        """
		This function computes the I_s4 term of the noise covariance, for bins defined by [rs[a1], rs[a1+1]] and
		[rs[a2], rs[a2+1]]. Those bins are expressed in arcmin. b and d are the redshift bins under consideration.
        """
		
        if a1 != a2 or b1 != b2:
            return 0
		
        else:
            return xi1_LOS_plus_intp(0)/(2*Omegas[a1]*n_b)

    scov = np.zeros((Nbin,Nbin))
        
    nsamples0 = load_file("nsample_dicts/LpLp/scov0") if use_measured_samples and os.path.exists("nsample_dicts/LpLp/scov0") else None
    nsamples1 = load_file("nsample_dicts/LpLp/scov1") if use_measured_samples and os.path.exists("nsample_dicts/LpLp/scov1") else None
    
    for a1 in range(Nbin):

        for a2 in range(Nbin):

            if b1 == b2:
        
                # Use preloaded nsamples or fall back to nsamp
                nsample0 = int(nsamples0[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples0 else nsamp
                nsample1 = int(nsamples1[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples1 else nsamp
                
                Is0, err0 = monte_carlo_integrate(func_s0, [(rs[a1], rs[a1+1]), (0, 2*np.pi), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample0)            
                Is1, err1 = monte_carlo_integrate(func_s1, [(rs[a1], rs[a1+1]), (0, 2*np.pi), (rs[a2], rs[a2+1]), (0, 2*np.pi)], nsample1) 
            else:
                Is0 = 0
                Is1 = 0
                err0 = 0
                err1 = 0
            
            Is4 = I_s4(a1, a2)

            scov[a1][a2] = Is0 + (Is1+Is4)/Nlens
            err = err0**2+ err1**2 / Nlens**2
            
            if not approx:

                Is2, err2 = monte_carlo_integrate(func_s2, [(rs[a1], rs[a1+1]), (rs[a2], rs[a2+1])], int(7e3))
                Is3, err3 = I_s3(a1, a2)

                scov[a1][a2] +=  (-Is0+Is3)/Nlens
                err += (err2**2 + err3**2) / Nlens**2

            err = np.sqrt(err)
            
            test_err(err, scov[a1, a2], f'LpLp scov redshifts {b1, b2} angular bins {a1, a2}')
            
    
    return scov, None, None, None, None
