import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from galaxy_distribution import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('xi2_LOS_plus_intp','xi1_LOS_plus_intp','xi32_LOS_plus_intp','xi2_LOS_minus_intp','xi1_LOS_minus_intp','xi32_LOS_minus_intp')
get_item('xi2_eps_plus_intp','xi1_eps_plus_intp','xi2_eps_minus_intp','xi1_eps_minus_intp')
get_item('xi2_d_intp')
get_item('xi2_LOS_eps_plus_intp', 'xi2_LOS_eps_minus_intp', 'xi32_LOS_eps2_plus_intp', 'xi32_LOS_eps2_minus_intp', 'xi32_LOS2_eps_plus_intp', 'xi32_LOS2_eps_minus_intp')
get_item('xi2_LOS_d_intp', 'xi1_LOS_d_intp', 'xi32_LOS_d2_intp', 'xi32_LOS2_d_intp')
get_item('xi2_d_eps_intp', 'xi1_d_eps_intp', 'xi32_d_eps2_intp', 'xi32_d2_eps_intp')

############################################################ 6.6 LpLe ########################################################################
##############################################################################################################################################

    
################################################# 6.6.1 LpLe noise covariance ################################################################

def generate_ncov_LpLe(distributions, sigma_noise, sigma_shape, b1, b2, approx=False):
    """
    Computes the contribution of noise in the covariance matrix
    of the LOS shear - ellipticity x LOS shear - galaxy position 
    correlation functions.
    
    distributions  : statistics relating to the distributions of lenses and galaxies
    sigma_noise    : the noise on the lens measurements
    sigma_shape    : the intrinsic galaxy shapes
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['Lp']
    distribution2 = distributions['Le']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for Lp 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for Lp (in rad)
    
    Nbin2       = distribution2.Nbina     #the number of angular bins for Le 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Le (in rad)
        
    nsamples_p = load_file("nsample_dicts/LpLe/ncovp") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/ncovp") else None
    nsamples_m = load_file("nsample_dicts/LpLe/ncovm") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/ncovm") else None
		
    def I_s4(a1, a2, eta, nsampl):
        """
        This function computes the term in the noise covariance (equal to the fourth term of sparsity covariance), 
        for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
        """

        def integrand_s4(params):

            phi_l, l, phi_g, g = params
            
            Theta_gl = np.sqrt( g**2 + l**2 - 2*g*l*np.cos(phi_g-phi_l) )

            f = np.cos(2*(phi_g-phi_l+eta*phi_l)) * xi2_d_eps_intp[b1][b2](Theta_gl) *g*l

            return f

        integral, err = monte_carlo_integrate(integrand_s4, [(0, 2*np.pi), (rs2[a2], rs2[a2+1]), (0, 2*np.pi), (rs1[a1], rs1[a1+1])], nsampl)

        integral /= ( Omegas1[a1]*Omegas2[a2] )
        err /= ( Omegas1[a1]*Omegas2[a2] )

        return integral, err

    ncov_p = np.zeros((Nbin1,Nbin2))
    ncov_m = np.zeros((Nbin1,Nbin2)) 
    
    for a1 in range(Nbin1):

        for a2 in range(Nbin2): 
        
            # Use preloaded nsamples or fall back to nsamp
            nsample_p = int(nsamples_p[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_p else nsamp
            nsample_m = int(nsamples_p[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_p else nsamp

            Is4p, errp = I_s4(a1, a2, 1, nsample_p)
            Is4m, errm = I_s4(a1, a2, -1, nsample_m)
            
            ncov_p[a1, a2] = -Is4p*sigma_noise**2/(2*Nlens)
            ncov_m[a1, a2] = -Is4m*sigma_noise**2/(2*Nlens)

            errp = errp*sigma_noise**2/(2*Nlens)
            errm = errm*sigma_noise**2/(2*Nlens)

            test_err(errp, ncov_p[a1, a2], f'LpLe ncov p redshifts {b1, b2} angular bins {a1, a2}')
            test_err(errm, ncov_m[a1, a2], f'LpLe ncov m redshifts {b1, b2} angular bins {a1, a2}')

    ncov = np.block([[ncov_p],
                     [ncov_m]])

    return ncov, ncov_p, ncov_m, None, None


################################################# 6.6.2 LpLe cosmic covariance ###############################################################

def generate_ccov_LpLe(distributions, b1, b2, approx=False):
    """
    Computes the contribution of cosmic variance in the covariance matrix
    of the LOS shear - ellipticity x LOS shear - galaxy position 
    correlation functions.
    
    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['Lp']
    distribution2 = distributions['Le']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for Lp 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for Lp (in rad)
    
    Nbin2       = distribution2.Nbina     #the number of angular bins for Le 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Le (in rad)
        
    nsamples_p1 = load_file("nsample_dicts/LpLe/ccovp_Ic1") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/ccovp_Ic1") else None
    nsamples_p2 = load_file("nsample_dicts/LpLe/ccovp_Ic2") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/ccovp_Ic2") else None
    nsamples_m1 = load_file("nsample_dicts/LpLe/ccovm_Ic1") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/ccovm_Ic1") else None
    nsamples_m2 = load_file("nsample_dicts/LpLe/ccovm_Ic2") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/ccovm_Ic2") else None

    def cosmic(a1, a2, eta):
        
        def func_c1(params):
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
            f = k*g*h*(xi2_LOS_eps_plus_intp[b2](Theta_h)*np.cos(2*(psi_kh-phi_g-eta*psi_kh)) 
                + xi2_LOS_eps_minus_intp[b2](Theta_h)*np.cos(2*(2*phi_h - phi_g - psi_kh + eta*psi_kh)) )*xi2_LOS_d_intp[b1](Theta_kg)
            prefactor = -np.pi/( Omegatot*Omegas1[a1]*Omegas2[a2] )
            
            return f*prefactor
            
            
        def func_c2(params):
            # unpack the parameters
            k, g, phi_g, h, psi_kh = params 
            # we need to compute additional angles that are not directly integration variables
            Theta_gh = np.sqrt( g**2 + h**2 + k**2 + 2*k*(h*np.cos(psi_kh)-g*np.cos(phi_g))- 2*g*h*np.cos(phi_g-psi_kh) )
            Theta_h = np.sqrt( h**2 + k**2 + 2*h*k*np.cos(psi_kh) )
            arg = (k+h*np.cos(psi_kh))/Theta_h
            # because of numeric instability I must make sure the result is between -1 and 1, as I will apply the arccos
            arg = np.clip(arg, -1, 1)
            # use of arccos
            phi_h = np.arccos(arg)
            phi_h[psi_kh > np.pi] *= -1 #  because of how arccos works; it returns a value between 0 and pi, I must correct for that
        
            # final calculation
            f = k*g*h*( xi2_LOS_plus_intp(k)*np.cos(2*(psi_kh-phi_g-eta*psi_kh))
                   + xi2_LOS_minus_intp(k)*np.cos(2*(2*phi_h - phi_g - psi_kh + eta*psi_kh)) )*xi2_d_eps_intp[b1][b2](Theta_gh)
            prefactor = -np.pi/( Omegatot*Omegas1[a1]*Omegas2[a2] )
            
            return f*prefactor
        
        # Use preloaded nsamples or fall back to nsamp
        if eta == 1:
            nsample1 = int(nsamples_p1[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_p1 else nsamp
            nsample2 = int(nsamples_p2[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_p2 else nsamp
        else:
            nsample1 = int(nsamples_m1[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_m1 else nsamp
            nsample2 = int(nsamples_m2[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_m2 else nsamp

        Ic1, err1 = monte_carlo_integrate(func_c1, [(0, r2_max), (rs1[a1], rs1[a1+1]), (0, 2*np.pi), (rs2[a2], rs2[a2+1]), (0, 2*np.pi)], nsample1)

        Ic2, err2 = monte_carlo_integrate(func_c2, [(0, r2_max), (rs1[a1], rs1[a1+1]), (0, 2*np.pi), (rs2[a2], rs2[a2+1]), (0, 2*np.pi)], nsample2)
        
        return Ic1 + Ic2, np.sqrt(err1**2+err2**2)

    ccov_p = np.zeros((Nbin1, Nbin2))
    ccov_m = np.zeros((Nbin1, Nbin2))
    
    for a1 in range(Nbin1):

        for a2 in range(Nbin2): # this could be made more efficient if the matrix is symmetric (dunno)
            
            ccov_p[a1, a2], errp = cosmic(a1,a2, 1)
            ccov_m[a1, a2], errm = cosmic(a1,a2, -1)

            test_err(errp, ccov_p[a1, a2], f'LpLe ccov p redshifts {b1, b2} angular bins {a1, a2}')
            test_err(errm, ccov_m[a1, a2], f'LpLe ccov m redshifts {b1, b2} angular bins {a1, a2}')

    ccov = np.block([[ccov_p],
                     [ccov_m]])

    return ccov, None, None, None, None

################################################# 6.6.3 LpLe sparsity covariance #############################################################

def generate_scov_LpLe(distributions, b1, b2, approx=False):
    """
    Computes the contribution of sparsity in the covariance matrix
    of the LOS shear - ellipticity x LOS shear - galaxy position 
    correlation functions.
    
    distributions  : statistics relating to the distributions of lenses and galaxies
    b1             : the galaxy redshift bin b  (0 to 4)
    b2             : the galaxy redshift bin b' (0 to 4)
    approx         : Bool, True to give the approximate result (quicker)
    """
    
    distribution1 = distributions['Lp']
    distribution2 = distributions['Le']
    
    Omegatot    = distribution1.Omegatot   #\Omega in the math - the total solid angle covered by the survey (should be the same for lenses and galaxies)
    Nbin1       = distribution1.Nbina      #the number of angular bins for Lp 
    Omegas1     = distribution1.Omegas     #\Omega_a in the math - the solid angle of bin a (in rad^2)
    rs1         = distribution1.limits     #the angular bin limits for Lp (in rad)
    
    Nbin2       = distribution2.Nbina     #the number of angular bins for Le 
    Omegas2     = distribution2.Omegas     #\Omega_a' in the math - the solid angle of bin a' (in rad^2)
    rs2         = distribution2.limits     #the angular bin limits for Le (in rad)
        
    nsamples_p3 = load_file("nsample_dicts/LpLe/scovp_Is3") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/scovp_Is3") else None
    nsamples_p4 = load_file("nsample_dicts/LpLe/scovp_Is4") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/scovp_Is4") else None
    nsamples_m3 = load_file("nsample_dicts/LpLe/scovm_Is3") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/scovm_Is3") else None
    nsamples_m4 = load_file("nsample_dicts/LpLe/scovm_Is4") if use_measured_samples and os.path.exists("nsample_dicts/LpLe/scovm_Is4") else None

    def sparsity(a1, a2, eta):

        def I_s1(a1, a2, eta):
            """
            This function computes the first term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
            """

            def integrand_s11(l):
                return xi32_LOS_d2_intp[b1](l) * 2 * l 

            def integrand_s12(l):
                if eta > 0:
                    return xi32_LOS_eps2_plus_intp[b2](l) * 2 * l
                else:
                    return xi32_LOS_eps2_minus_intp[b2](l) * 2 * l

            integral1, err1 = integrate.quad(integrand_s11, rs1[a1], rs1[a1+1])
            integral2, err2 = integrate.quad(integrand_s12, rs2[a2], rs2[a2+1])

            result = (np.pi**2) * integral1 * integral2 / (Omegas1[a1] * Omegas2[a2])
            err = np.sqrt(((np.pi**2) * err1 * integral2 / (Omegas1[a1] * Omegas2[a2]))**2 + ((np.pi**2) * integral1 * err2 / (Omegas1[a1] * Omegas2[a2]))**2)
            
            return result, err

        def I_s2(a1, a2, eta):
            """
            This function computes the second term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
            """

            def integrand_s21(l):
                return xi2_LOS_d_intp[b1](l) * 2 * l

            def integrand_s22(l):
                if eta > 0:
                    return xi2_LOS_eps_plus_intp[b2](l) * 2 * l
                else:
                    return xi2_LOS_eps_minus_intp[b2](l) * 2 * l

            integral1, err1 = integrate.quad(integrand_s21, rs1[a1], rs1[a1+1])
            integral2, err2 = integrate.quad(integrand_s22, rs2[a2], rs2[a2+1])

            results = - 2*(np.pi**2) * integral1 * integral2 / (Omegas1[a1] * Omegas2[a2])
            err = np.sqrt((2*(np.pi**2) * err1 * integral2 / (Omegas1[a1] * Omegas2[a2]))**2 + (2*(np.pi**2) * integral1 * err2 / (Omegas1[a1] * Omegas2[a2]))**2)

            return results, err

        def func_s3(params):
            # unpack the parameters
            g, phi_g, h, phi_h = params

            f = g * h * (xi32_LOS_eps2_plus_intp[b2](h) * np.cos(2 * (phi_h - phi_g - eta * phi_h))
				+ xi32_LOS_eps2_minus_intp[b2](h) * np.cos(2 * (phi_h - phi_g + eta * phi_h))) * xi32_LOS_d2_intp[b1](g)
            prefactor = 1 / (2 * Omegas1[a1] * Omegas2[a2])

            return f * prefactor

        def I_s4(a1, a2, eta, nsampl):
            """
            This function computes the fourth term in the sparsity covariance, for bins defined by [rs[a1], rs[a1+1] and [rs[a2], rs[a2+1]]. Those bins are expressed in arcmin.
            """
            def integrand_s4(params):

                phi_l, l, phi_g, g = params
                Theta_gl = np.sqrt(g**2 + l**2 - 2 * g * l * np.cos(phi_g - phi_l))
                f = np.cos(2 * (phi_g - phi_l + eta * phi_l)) * xi2_d_eps_intp[b1][b2](Theta_gl) * g * l #slightly confusing switch between phi_l and phi_h here, but I think it's all consistent
                return f

            integral, err = monte_carlo_integrate(integrand_s4, [(0, 2 * np.pi), (rs2[a2], rs2[a2+1]), (0, 2 * np.pi), (rs1[a1], rs1[a1+1])], nsampl)

            result = integral * xi1_LOS_plus_intp(0) / (2 * Omegas1[a1] * Omegas2[a2] )
            err = err * xi1_LOS_plus_intp(0) / (2 * Omegas1[a1] * Omegas2[a2] )
            return result, err
        
        # Use preloaded nsamples or fall back to nsamp
        if eta == 1:
            nsample3 = int(nsamples_p3[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_p3 else nsamp
            nsample4 = int(nsamples_p4[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_p4 else nsamp
        else:
            nsample3 = int(nsamples_m3[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_m3 else nsamp
            nsample4 = int(nsamples_m4[str(b1)+str(b2)][str(a1) + str(a2)]) if nsamples_m4 else nsamp

        Is1, err1 = I_s1(a1, a2, eta)
        Is2, err2 = I_s2(a1, a2, eta)
        Is4, err4 = I_s4(a1, a2, eta, nsample4)

        if approx:
            Is3 = 0
            err3 = 0
        else:
            Is3, err3 = monte_carlo_integrate(func_s3, [(rs1[a1], rs1[a1+1]), (0, 2 * np.pi), (rs2[a2], rs2[a2+1]), (0, 2 * np.pi)], nsample3)

        result = -(Is1 + Is2 + Is3 + Is4) / Nlens
        err = np.sqrt(err1**2 + err2**2 + err3**2 + err4**2) / Nlens
        
        return result, err

    scov_p = np.zeros((Nbin1, Nbin2))
    scov_m = np.zeros((Nbin1, Nbin2))
    
    for a1 in range(Nbin1):
        for a2 in range(Nbin2):
            scov_p[a1, a2], errp = sparsity(a1, a2, 1)
            scov_m[a1, a2], errm = sparsity(a1, a2, -1)

            test_err(errp, scov_p[a1, a2], f'LpLe scov p redshifts {b1, b2} angular bins {a1, a2}')
            test_err(errm, scov_m[a1, a2], f'LpLe scov m redshifts {b1, b2} angular bins {a1, a2}')
            

    scov = np.block([[scov_p],
					 [scov_m]])

    return scov, scov_p, scov_m, None, None