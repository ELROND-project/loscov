from config import * 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from useful_functions import *

get_item('chid_Euclid', 'chis_Euclid')

################################################ LOS Weight function #####################################################

def K_LOS(chi, chid, chis):
    """
    LOS weight function

    chi : an inputted value of comoving distance
    chid : the comoving distance to the lens
    chis : the comoving distance to the source
    """
        
    os = (chis - chi) / chis #the weight function for gamma_os
    od = (chid - chi) / chid #the weight function for gamma_od
    ds = (chi - chid) * (chis - chi) / (chi * (chis - chid) ) #the weight function for gamma_ds

    #the actual weight function
    K  = (os * np.heaviside(os, 0)    #returns 0 if os is negative
          + od * np.heaviside(od, 0)
          - ds * np.heaviside(ds, 0))
    
    return K

def K_LOS_mean(chi):
    """
    Redshift-averaged LOS weight function
    
    chi : an inputted comoving distance
    """
    K = np.mean(K_LOS(chi, chid_Euclid, chis_Euclid))
    
    return K

def KK_LOS_mean(chi):
    """
    Redshift-averaged LOS weight function
    
    chi : an inputted comoving distance
    """
    K = np.mean(K_LOS(chi, chid_Euclid, chis_Euclid) * K_LOS(chi, chid_Euclid, chis_Euclid))
    
    return K

def Q_LOS_mean(chi):
    """
    Redshift-averaged LOS integration kernel
    
    chi : an inputted comoving distance
    """
    redshift = background.redshift_at_comoving_radial_distance(chi)
    
    Q = -1.5 * Omega_M * (H0/(c*1e-3))**2 * (1+redshift) * K_LOS_mean(chi)
    
    return Q

def QQ_LOS_mean(chi):
    """
    Redshift-averaged LOS integration kernel
    
    chi : an inputted comoving distance
    """
    redshift = background.redshift_at_comoving_radial_distance(chi)
    
    QQ = ( 1.5 * Omega_M * (H0/(c*1e-3))**2 )**2 * (1+redshift) * KK_LOS_mean(chi)
    
    return QQ


################################################ Getting cls #####################################################

def get_cl_L(chimax, lmax, nl):
    """
    This function generates Cls for convergence and shear
    It takes as argument the maximum multipole lmax.
    nl is the number of values to be computed.
    """

    get_item('Q_LOS_mean_intp', 'QQ_LOS_rms_intp')
    
    nz = 100 #number of elements for discrete integral along the los
    
    # Conformal distances and redshifts
    results = camb.get_background(pars)
    chis = np.linspace(0, chimax, nz)
    zs = results.redshift_at_comoving_radial_distance(chis)
    
    # Array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #CAMB correction
    CAMB_factor = ((1+zs) * 1.5 * Omega_M * (H0/(c*1e-3))**2)**(-1)
    
    # Lensing kernel (here LOS shear)
    kernel2 = Q_LOS_mean_intp(chis)**2 * CAMB_factor**2 
    kernel1 = QQ_LOS_rms_intp(chis)**2 * CAMB_factor**2 
    
    # Integration over chi
    lmin = 1
    ls = np.logspace(np.log10(lmin), np.log10(lmax), nl)
    cl2 = np.zeros(ls.shape)
    cl1 = np.zeros(ls.shape)
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
        
    return ls, cl2, cl1


