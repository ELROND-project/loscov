import numpy as np
import config_lite as config
from functions import useful_functions as uf 
from cosmology import background
from scipy.interpolate import CubicSpline
from itertools import product

if config.compute_correlations:

    Thetamin = uf.arcmintorad(config.Thetamin_arcmin)  #minimum theta from which we calculate correlation functions (in radians)
    
    ####################################### 1.2 Euclid lenses and galaxies ##################################################
    #########################################################################################################################
    
    #reading in the forecasted sample of Euclid lenses
    Euclid_lenses = np.loadtxt('lenses_Euclid.txt')
    zd = Euclid_lenses[:, 0]
    zs = Euclid_lenses[:, 1]
    
    # convert into comoving distances (in Mpc)
    chid = background.comoving_radial_distance(zd)
    chis = background.comoving_radial_distance(zs)
    
    chimax_lens = max(chis)

    zmax_gal = max(config.binparams['redshifts']) 

    chimax_gal = background.comoving_radial_distance(zmax_gal)

    chimax = max(chimax_lens,chimax_gal) 
    
    #place these variables in the global dictionary
    uf.add_dict(chimax, chid, chis, zd, zs)

    ##############################################################################################################################
    ############################################# 2. AUTOCORRELATION FUNCTIONS ###################################################
    ##############################################################################################################################
    
    from functions.correlations.get_correlations import get_correlations, get_DD_correlations, get_gD_correlations

    lmax = config.lmax
    nl = config.nl
    Thetamax = config.Thetamax
    nTheta = config.nTheta
    
    ######################################################## 2.1 Shear ###########################################################
    ##############################################################################################################################
    
    from functions.correlations import shear
    
    ################################################# 2.1.1 weight functions ######################################################
    
    # Interpolate to get a fast 1D weight function
    W_LOS_mean_vec = np.vectorize(shear.W_LOS_mean)
    chi = np.linspace(0, chimax, 100)
    W = W_LOS_mean_vec(chi)
    W_LOS_mean_intp = CubicSpline(chi, W)
    
    # Interpolate to get a fast 1D weight function
    WW_LOS_mean_vec = np.vectorize(shear.WW_LOS_mean)
    chi = np.linspace(0, chimax, 100)
    WW = WW_LOS_mean_vec(chi)
    WW_rms = np.sqrt(WW)
    WW_LOS_rms_intp = CubicSpline(chi, WW_rms)
    
    uf.add_dict(W_LOS_mean_intp, WW_LOS_rms_intp)
    
    ######################################################## 2.1.2 cls ###########################################################
    
    ls, cl2, cl1, cl32 = shear.get_cls_gamma_LOS(chimax, lmax, nl)
    cl2_LOS_intp = CubicSpline(ls, cl2)
    cl1_LOS_intp = CubicSpline(ls, cl1)
    cl32_LOS_intp = CubicSpline(ls, cl32)
    
    ############################################# 2.1.3 correlation functions ####################################################

    Theta, xi2_LOS_plus, xi2_LOS_minus = get_correlations(
        cl2_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    Theta, xi1_LOS_plus, xi1_LOS_minus = get_correlations(
        cl1_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    Theta, xi32_LOS_plus, xi32_LOS_minus = get_correlations(
        cl32_LOS_intp, Thetamin, Thetamax, nTheta=nTheta)
    
    Theta_arcmin = uf.radtoarcmin(Theta)
    
    xi2_LOS_plus_intp = CubicSpline(Theta, xi2_LOS_plus)
    xi1_LOS_plus_intp = CubicSpline(Theta, xi1_LOS_plus)
    xi32_LOS_plus_intp = CubicSpline(Theta, xi32_LOS_plus)
    xi2_LOS_minus_intp = CubicSpline(Theta, xi2_LOS_minus)
    xi1_LOS_minus_intp = CubicSpline(Theta, xi1_LOS_minus)
    xi32_LOS_minus_intp = CubicSpline(Theta, xi32_LOS_minus)
    
    print('Finished 2.1 LOS autocorrelation functions')
    
    uf.add_dict(xi2_LOS_plus_intp, xi1_LOS_plus_intp, xi32_LOS_plus_intp, xi2_LOS_minus_intp, xi1_LOS_minus_intp, xi32_LOS_minus_intp)
    
    ######################################################## 2.2 Shape ###########################################################
    ##############################################################################################################################
    

    from functions.correlations import shape
    
    ################################################# 2.2.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    W_os_mean_intp = []
    
    for b in range(5): # NH: what are these hardcoded 5s?
        W_os_mean_vec = np.vectorize(shape.W_os_mean)
        chi = np.linspace(1e-3, chimax, 100)
        W = W_os_mean_vec(chi, b)
        W_os_mean_intp.append(CubicSpline(chi, W))
    
    # Interpolate to get fast 1D weight functions
    
    WW_os_rms_intp = []
    
    for b in range(5):
        WW_os_mean_vec = np.vectorize(shape.WW_os_mean)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        WW = WW_os_mean_vec(chi, b)
        WW_rms = np.sqrt(WW)                             #potential for confusion - WW_rms is actually order W
        WW_os_rms_intp.append(CubicSpline(chi, WW_rms))
    
    uf.add_dict(W_os_mean_intp, WW_os_rms_intp)
    
    ######################################################## 2.2.2 cls ###########################################################
    
    ls_list = []
    cl2_eps_intp = []
    cl1_eps_intp = []
    
    for b1 in range(5): #loop through b
    
        cl2_eps_intp.append([])
    
        for b2 in range(5): #loop through b'
            
            ls, cl2, cl1 = shape.get_cl_gamma(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_eps_intp[b1].append(CubicSpline(ls, cl2))
    
        ls_list.append(ls)
        cl1_eps_intp.append(CubicSpline(ls, cl1))
    
    ############################################# 2.2.3 correlation functions ####################################################
    
    Theta_list = []
    xi2_eps_plus_list = []
    xi2_eps_minus_list = []
    xi1_eps_plus_list = []
    xi1_eps_minus_list = []
    
    for b1 in range(5):
            
        xi2_eps_plus_list.append([])
        xi2_eps_minus_list.append([])
        
        Theta, xi1_eps_plus, xi1_eps_minus = get_correlations(
            cl1_eps_intp[b1], Thetamin, Thetamax, nTheta)
    
        for b2 in range(5):
            Theta, xi2_eps_plus, xi2_eps_minus = get_correlations(
                cl2_eps_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            xi2_eps_plus_list[b1].append(xi2_eps_plus)
            xi2_eps_minus_list[b1].append(xi2_eps_minus)
        
        Theta_list.append(Theta)
        xi1_eps_plus_list.append(xi1_eps_plus)
        xi1_eps_minus_list.append(xi1_eps_minus)
    
    xi2_eps_plus_intp = []
    xi1_eps_plus_intp = []
    xi2_eps_minus_intp = []
    xi1_eps_minus_intp = []
    
    for b1 in range(5):
        
        xi1_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi1_eps_plus_list[b1]))
        xi1_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi1_eps_minus_list[b1]))
        xi2_eps_plus_intp.append([])
        xi2_eps_minus_intp.append([])
    
        for b2 in range(5):
            xi2_eps_plus_intp[b1].append(CubicSpline(Theta_list[b1], xi2_eps_plus_list[b1][b2]))
            xi2_eps_minus_intp[b1].append(CubicSpline(Theta_list[b1], xi2_eps_minus_list[b1][b2]))
            
    print('Finished 2.2 shape autocorrelation functions')
    
    uf.add_dict(xi2_eps_plus_intp,xi1_eps_plus_intp,xi2_eps_minus_intp,xi1_eps_minus_intp)
    
    ######################################################## 2.3 Position ########################################################
    ##############################################################################################################################

    from functions.correlations import position
    
    ################################################# 2.3.1 weight functions ######################################################
    
    # Interpolate to get fast 1D weight functions
    
    W_d_mean_intp = []
    
    for b in range(5):
        W_d_mean_vec = np.vectorize(position.W_d)
        chi = np.linspace(1e-3, chimax, 100)
        W = W_d_mean_vec(chi, b)
        W_d_mean_intp.append(CubicSpline(chi, W))
    
    # Interpolate to get fast 1D weight functions
    
    WW_d_rms_intp = []
    
    for b in range(5):
        WW_d_mean_vec = np.vectorize(position.WW_d)
        chi = np.linspace(1e-5, chimax, 100)                    #maybe parameterise these?
        WW = WW_d_mean_vec(chi, b)
        WW_rms = np.sqrt(WW)
        WW_d_rms_intp.append(CubicSpline(chi, WW_rms))

    W_d_intp = W_d_mean_intp  #redundant, fix this
    uf.add_dict(W_d_mean_intp, WW_d_rms_intp)
    uf.add_dict(W_d_intp)
    
    ######################################################## 2.3.2 cls ###########################################################
    
    ls_list = []
    cl1_d_intp = []
    cl2_d_intp = []
    cl32_d_intp = []
    
    for b1 in range(5): #loop through b
    
        cl2_d_intp.append([])
        cl32_d_intp.append([])
        
        for b2 in range(5): #loop through b'
            
            ls, cl2, cl1, cl32 = position.get_cl_d(b1, b2, chimax, lmax, nl) #each time we recalculate everything (a bit inefficient)
            
            cl2_d_intp[b1].append(CubicSpline(ls, cl2))
            cl32_d_intp[b1].append(CubicSpline(ls, cl32))     #almost certainly wrong
        
        ls_list.append(ls)
        cl1_d_intp.append(CubicSpline(ls, cl1)) 

    #Note - because of the weight function, all of these will be zero unless b1 == b2 
    
    ############################################# 2.3.3 correlation functions ####################################################
    
    Theta_list = []
    xi2_d_list = []
    xi1_d_list = []
    xi32_d_list = []
    
    for b1 in range(5):
            
        xi2_d_list.append([])
        xi32_d_list.append([])
            
        Theta, xi1_d = get_DD_correlations(
            cl1_d_intp[b1], Thetamin, Thetamax, nTheta)
        
        Theta_list.append(Theta)
        xi1_d_list.append(xi1_d)
    
        for b2 in range(5):
            Theta, xi2_d = get_DD_correlations(
                cl2_d_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            xi2_d_list[b1].append(xi2_d)
            
            Theta, xi32_d = get_DD_correlations(
                cl32_d_intp[b1][b2], Thetamin, Thetamax, nTheta)
            
            xi32_d_list[b1].append(xi32_d)
    
    xi2_d_intp = []
    xi1_d_intp = []
    xi32_d_intp = []
    
    for b1 in range(5):
        
        xi2_d_intp.append([])
        xi32_d_intp.append([])
        xi1_d_intp.append(CubicSpline(Theta_list[b1], xi1_d_list[b1]))
    
        for b2 in range(5):
            xi2_d_intp[b1].append(CubicSpline(Theta_list[b1], xi2_d_list[b1][b2]))
            xi32_d_intp[b1].append(CubicSpline(Theta_list[b1], xi32_d_list[b1][b2]))
            
    print('Finished 2.3 position autocorrelation functions')
    
    uf.add_dict(xi2_d_intp, xi1_d_intp, xi32_d_intp)
    
    ##############################################################################################################################
    ############################################ 3. MIXED CORRELATION FUNCTIONS ##################################################
    ##############################################################################################################################
    
    ##################################################### 3.1 shear shape ########################################################
    ##############################################################################################################################
    
    from functions.correlations import shear_shape
    
    ls_list = []
    cl2LOSos_intp_list = []
    cl32LOSos2_intp_list = []
    cl32LOS2os_intp_list = []
    cl1LOSos_intp_list = []
    
    for b in range(5):

        ls, cl2LOSos, cl32LOSos2, cl32LOS2os, cl1LOSos = shear_shape.get_cls_mixed_LOS_os(b, chimax, lmax, nl)

        ls_list.append(ls)

        cl2LOSos_intp_list.append(CubicSpline(ls, cl2LOSos))
        cl32LOSos2_intp_list.append(CubicSpline(ls, cl32LOSos2))
        cl32LOS2os_intp_list.append(CubicSpline(ls, cl32LOS2os))
        cl1LOSos_intp_list.append(CubicSpline(ls, cl1LOSos))
    
    Theta_list = []
    
    xi2_LOS_eps_plus_list = []
    xi2_LOS_eps_minus_list = []
    
    xi32_LOS_eps2_plus_list = []
    xi32_LOS_eps2_minus_list = []
    
    xi32_LOS2_eps_plus_list = []
    xi32_LOS2_eps_minus_list = []
    
    xi1_LOS_eps_plus_list = []
    xi1_LOS_eps_minus_list = []
    
    for b1 in range(5):
        
        Theta, xi2_LOS_eps_plus, xi2_LOS_eps_minus = get_correlations(
            cl2LOSos_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi2_LOS_eps_plus_list.append(xi2_LOS_eps_plus)
        xi2_LOS_eps_minus_list.append(xi2_LOS_eps_minus)
        
        Theta, xi32_LOS_eps2_plus, xi32_LOS_eps2_minus = get_correlations(
            cl32LOSos2_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi32_LOS_eps2_plus_list.append(xi32_LOS_eps2_plus)
        xi32_LOS_eps2_minus_list.append(xi32_LOS_eps2_minus)
        
        Theta, xi32_LOS2_eps_plus, xi32_LOS2_eps_minus = get_correlations(
            cl32LOS2os_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi32_LOS2_eps_plus_list.append(xi32_LOS2_eps_plus)
        xi32_LOS2_eps_minus_list.append(xi32_LOS2_eps_minus)
        
        Theta, xi1_LOS_eps_plus, xi1_LOS_eps_minus = get_correlations(
            cl1LOSos_intp_list[b1], Thetamin, Thetamax, nTheta)
        
        xi1_LOS_eps_plus_list.append(xi1_LOS_eps_plus)
        xi1_LOS_eps_minus_list.append(xi1_LOS_eps_minus)
    
        Theta_list.append(Theta)
    
    xi2_LOS_eps_plus_intp = []
    xi2_LOS_eps_minus_intp = []
    
    xi32_LOS_eps2_plus_intp = []
    xi32_LOS_eps2_minus_intp = []
    
    xi32_LOS2_eps_plus_intp = []
    xi32_LOS2_eps_minus_intp = []
    
    xi1_LOS_eps_plus_intp = []
    xi1_LOS_eps_minus_intp = []
    
    for b1 in range(5):
        
        xi2_LOS_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi2_LOS_eps_plus_list[b1]))
        xi2_LOS_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi2_LOS_eps_minus_list[b1]))
            
        xi32_LOS_eps2_plus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS_eps2_plus_list[b1]))
        xi32_LOS_eps2_minus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS_eps2_minus_list[b1]))
        
        xi32_LOS2_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS2_eps_plus_list[b1]))
        xi32_LOS2_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi32_LOS2_eps_minus_list[b1]))
            
        xi1_LOS_eps_plus_intp.append(CubicSpline(Theta_list[b1], xi1_LOS_eps_plus_list[b1]))
        xi1_LOS_eps_minus_intp.append(CubicSpline(Theta_list[b1], xi1_LOS_eps_minus_list[b1]))
    
    print('Finished 3.1 shear shape correlation functions')
    
    uf.add_dict(xi2_LOS_eps_plus_intp, xi2_LOS_eps_minus_intp, xi32_LOS_eps2_plus_intp, xi32_LOS_eps2_minus_intp, xi32_LOS2_eps_plus_intp, xi32_LOS2_eps_minus_intp, xi1_LOS_eps_plus_intp, xi1_LOS_eps_minus_intp)
    
    #################################################### 3.2 shear position ######################################################
    ##############################################################################################################################
    
    from functions.correlations import shear_position
    
    ls_list = []
    cl2LOSd_intp_list = []
    cl32LOSd2_intp_list = []
    cl32LOS2d_intp_list = []
    cl1LOSd_intp_list = []
    
    for b in range(5):
        
        ls, cl2LOSd, cl32LOSd2, cl32LOS2d, cl1LOSd = shear_position.get_cls_mixed_LOS_d(b, chimax, lmax, nl)
            
        ls_list.append(ls)
        cl2LOSd_intp_list.append(CubicSpline(ls, cl2LOSd))
        cl32LOS2d_intp_list.append(CubicSpline(ls, cl32LOS2d))
        cl32LOSd2_intp_list.append(CubicSpline(ls, cl32LOSd2))
        cl1LOSd_intp_list.append(CubicSpline(ls, cl1LOSd))
    
    Theta_list = []
    
    xi2_LOS_d_list = []
    
    xi32_LOS_d2_list = []
    
    xi32_LOS2_d_list = []
    
    xi1_LOS_d_list = []
    
    for b in range(5):
        
        Theta, xi2_LOS_d = get_gD_correlations(
            cl2LOSd_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi2_LOS_d_list.append(xi2_LOS_d)
        
        Theta, xi32_LOS2_d = get_gD_correlations(
            cl32LOS2d_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi32_LOS2_d_list.append(xi32_LOS2_d)
        
        Theta, xi32_LOS_d2 = get_gD_correlations(
            cl32LOSd2_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi32_LOS_d2_list.append(xi32_LOS_d2)
        
        Theta, xi1_LOS_d = get_gD_correlations(
            cl1LOSd_intp_list[b], Thetamin, Thetamax, nTheta)
        
        xi1_LOS_d_list.append(xi1_LOS_d)
    
        Theta_list.append(Theta)
    
    xi2_LOS_d_intp = []
    
    xi32_LOS_d2_intp = []
    
    xi32_LOS2_d_intp = []
    
    xi1_LOS_d_intp = []
    
    for b in range(5):
        
        xi2_LOS_d_intp.append(CubicSpline(Theta_list[b], xi2_LOS_d_list[b]))
        
        xi32_LOS2_d_intp.append(CubicSpline(Theta_list[b], xi32_LOS2_d_list[b]))
            
        xi32_LOS_d2_intp.append(CubicSpline(Theta_list[b], xi32_LOS_d2_list[b]))
            
        xi1_LOS_d_intp.append(CubicSpline(Theta_list[b], xi1_LOS_d_list[b]))
    
    print('Finished 3.2 shear position correlation functions')
    
    uf.add_dict(xi2_LOS_d_intp, xi32_LOS_d2_intp, xi32_LOS2_d_intp, xi1_LOS_d_intp)
    
    #################################################### 3.3 shape position ######################################################
    ##############################################################################################################################
    
    from functions.correlations import position_shape
    
    ls_list = []
    cl2dos_intp_list = []
    cl32dos2_intp_list = []
    cl32d2os_intp_list = []
    cl1dos_intp_list = []
    
    for b1 in range(5):
        
        cl2dos_intp_list.append([])
        cl32dos2_intp_list.append([])
        cl32d2os_intp_list.append([])
        cl1dos_intp_list.append([])
    
        for b2 in range(5):
        
            ls, cl2dos, cl32dos2, cl32d2os, cl1dos = position_shape.get_cls_mixed_d_os(b1, b2, chimax, lmax, nl)
            
            cl2dos_intp_list[b1].append(CubicSpline(ls, cl2dos))
            cl32dos2_intp_list[b1].append(CubicSpline(ls, cl32dos2))
            cl32d2os_intp_list[b1].append(CubicSpline(ls, cl32d2os))
            cl1dos_intp_list[b1].append(CubicSpline(ls, cl1dos))
            
        ls_list.append(ls)
    
    Theta_list = []
    
    xi2_d_eps_list = []
    
    xi32_d_eps2_list = []
    
    xi32_d2_eps_list = []
    
    xi1_d_eps_list = []
    
    for b1 in range(5):
        
        xi2_d_eps_list.append([])
        
        xi32_d_eps2_list.append([])
    
        xi32_d2_eps_list.append([])
    
        xi1_d_eps_list.append([])
        
        for b2 in range(5):
        
            Theta, xi2_d_eps_plus = get_gD_correlations(
                cl2dos_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi2_d_eps_list[b1].append(xi2_d_eps_plus)
        
            Theta, xi32_d_eps2_plus = get_gD_correlations(
                cl32dos2_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi32_d_eps2_list[b1].append(xi32_d_eps2_plus)
        
            Theta, xi32_d2_eps_plus = get_gD_correlations(
                cl32d2os_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi32_d2_eps_list[b1].append(xi32_d2_eps_plus)
        
            Theta, xi1_d_eps_plus = get_gD_correlations(
                cl1dos_intp_list[b1][b2], Thetamin, Thetamax, nTheta)
        
            xi1_d_eps_list[b1].append(xi1_d_eps_plus)
    
        Theta_list.append(Theta)
    
    xi2_d_eps_intp = []
    
    xi32_d_eps2_intp = []
    
    xi32_d2_eps_intp = []
    
    xi1_d_eps_intp = []
    
    for b1 in range(5):
        
        xi2_d_eps_intp.append([])
        xi32_d2_eps_intp.append([])
        xi32_d_eps2_intp.append([])
        xi1_d_eps_intp.append([])
    
        for b2 in range(5):
            
            xi2_d_eps_intp[b1].append(CubicSpline(Theta_list[b1], xi2_d_eps_list[b1][b2]))
        
            xi32_d_eps2_intp[b1].append(CubicSpline(Theta_list[b1], xi32_d_eps2_list[b1][b2]))
            
            xi32_d2_eps_intp[b1].append(CubicSpline(Theta_list[b1], xi32_d2_eps_list[b1][b2]))
            
            xi1_d_eps_intp[b1].append(CubicSpline(Theta_list[b1], xi1_d_eps_list[b1][b2]))
            
    print('Finished 3.3 shape position correlation functions')
    
    uf.add_dict(xi2_d_eps_intp, xi32_d2_eps_intp, xi32_d_eps2_intp, xi1_d_eps_intp)
        
    uf.save_pickle(config.global_dict, 'correlations.pkl', f"Saved all correlations")

##############################################################################################################################
############################################## 4 PREPARING FOR THE RUN #######################################################
##############################################################################################################################

############################################ 4.1 Defining distributions ######################################################
##############################################################################################################################

from functions.distribution import Distributions

Nlens = config.Nlens
NGal = config.NGal
Nbina_LL = config.Nbina_LL
Nbina_Le = config.Nbina_Le
Nbina_Lp = config.Nbina_Lp
Thetamax_LL = config.Thetamax_LL
Thetamax_Le = config.Thetamax_Le
Thetamax_Lp = config.Thetamax_Lp
sky_coverage = config.sky_coverage

distribution_LL = Distributions(Nlens, binscheme=config.binscheme_LL, sky_coverage=sky_coverage, Nbina=Nbina_LL, Thetamax=Thetamax_LL) 
distribution_Le = Distributions(NGal, binscheme=config.binscheme_Le, sky_coverage=sky_coverage, Nbina=Nbina_Le, Thetamax=Thetamax_Le)
distribution_Lp = Distributions(NGal, binscheme=config.binscheme_Lp, sky_coverage=sky_coverage, Nbina=Nbina_Lp, Thetamax=Thetamax_Lp)

distributions = {"LL": distribution_LL,
                "Le": distribution_Le,
                "Lp": distribution_Lp}

uf.save_pickle(distributions, 'distributions.pkl', f"Saved all distributions") # NH: add a file extension?

print(f"Successfully defined distribution functions.")

############################################### 4.2 Creating .txt file ########################$##############################
##############################################################################################################################

# Output file name
task_file = "tasks.txt"

b1_values = config.b1_values
b2_values = config.b2_values
cov_types = config.cov_types

print(b1_values)

# Open file to write task commands
with open(task_file, "w") as f:
    # Full (b1, b2) iteration
    for b1, b2 in product(b1_values, b2_values):
        for cov_matrix in config.cov_matrices_full:
            for cov_type in cov_types:
                f.write(f"{b1} {b2} {cov_matrix} {cov_type}\n")

    # Single b1 iteration
    for cov_matrix, b1 in product(config.cov_matrices_b1, b1_values):
        for cov_type in cov_types:
            f.write(f"{b1} None {cov_matrix} {cov_type}\n")

    # No b1, b2 iteration (only one call)
    for cov_matrix in config.cov_matrices_no_b:
        for cov_type in cov_types:
            f.write(f"None None {cov_matrix} {cov_type}\n")

print(f"Task file '{task_file}' created successfully with all required jobs.")
