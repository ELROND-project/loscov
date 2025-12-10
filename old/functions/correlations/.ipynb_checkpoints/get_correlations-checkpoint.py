##############################################################################################################################
########################################### 2. LOS AUTOCORRELATION FUNCTIONS #################################################
##############################################################################################################################

################################################ LOS Weight function #####################################################

def W_LOS(chi, chidd, chiss):
    """
    LOS weight function

    chi : an inputted value of comoving distance
    chidd : the comoving distance to the lens
    chiss : the comoving distance to the source
    """
    os = chi * (chiss - chi) / chis #the weight function for gamma_os
    od = chi * (chidd - chi) / chid #the weight function for gamma_od
    ds = (chi - chidd) * (chis - chi) / (chiss - chidd) #the weight function for gamma_ds

    #the actual weight function
    W  = (os * np.heaviside(os, 0)    #returns 0 if os is negative
          + od * np.heaviside(od, 0)
          - ds * np.heaviside(ds, 0))
    
    return W

def W_LOS_mean(chi):
    """
    Redshift-averaged LOS weight function (W_LOS,eff(chi) in the above)
    
    chi : an inputted comoving distance
    """
    W = np.mean(W_LOS(chi, chid, chis))
    
    return W

def WW_LOS_mean(chi):
    """
    Redshift-averaged LOS weight function squared

    chi : the comoving distance
    """
    WW = np.mean(W_LOS(chi, chid, chis) * W_LOS(chi, chid, chis))
    
    return WW

def get_cls_gamma_LOS(chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.
    """
    
    nz = 100 #number of elements for discrete integral along the los
    
    # Conformal distances and redshifts
    results = camb.get_background(pars)
    chis = np.linspace(0, chimax, nz)
    zs = results.redshift_at_comoving_radial_distance(chis)
    
    # Array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]
    
    # Lensing kernel (here LOS shear)
    kernel2 = W_LOS_mean_intp(chis)**2 / chis**2
    kernel1 = WW_LOS_rms_intp(chis)**2 / chis**2
    kernel32 = W_LOS_mean_intp(chis) * WW_LOS_rms_intp(chis) / chis**2
    
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    cl2 = np.zeros(ls.shape)
    cl1 = np.zeros(ls.shape)
    cl32 = np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    
    # Check that lmax is not too big
    if lmax > extrap_kmax * chis[-1]:
        print("""Warning: lmax is too large given the range of extrapolation given to CAMB
for the power spectrum. The results cannot be trusted.""")
    
    for i, l in enumerate(ls):
        k = (l + 0.5)/chis
        w[:] = 1
        w[k<1e-4] = 0
        w[k>=extrap_kmax] = 0
        cl2[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel2)
        cl1[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel1)
        cl32[i] = np.dot(dchis, w * Weyl_power_spectra.P(zs, k, grid=False) * kernel32)
        
    return ls, cl2, cl1, cl32

def get_correlations(cl_intp, Thetamin, Thetamax, nTheta=nTheta):
    """
    Generates the + and - correlation functions from the angular power spectrum
    using Hankel transformations, for a range of apertures [Thetamin, Thetamax]
    expressed in radians.
    The first argument is the interpolation of the cls.
    It returns the correlation functions as arrays.
    """
    
    # initialise Hankel transforms
    ht0 = HankelTransform(nu=0, N=1e4, h=1e-2) # for xi_plus
    ht4 = HankelTransform(nu=4, N=1e4, h=1e-2) # for xi_minus
    
    # define angular domain
    #logThetamin = np.log10(Thetamin)
    #logThetamax = np.log10(Thetamax)
    #nTheta = 100 * int(logThetamax - logThetamin) # 100 points per decade
    #Theta = np.logspace(logThetamin, logThetamax, nTheta) # separation angles, in arcmin
    logThetamin = np.log10(Thetamin)
    logThetamax = np.log10(Thetamax)
    Theta = np.logspace(logThetamin, logThetamax, nTheta)
    Theta = np.append([0], Theta)
    xi_plus = ht0.transform(cl_intp, Theta, ret_err=False)/(2*np.pi)
    xi_minus = ht4.transform(cl_intp, Theta, ret_err=False)/(2*np.pi)
    
    return Theta, xi_plus, xi_minus



