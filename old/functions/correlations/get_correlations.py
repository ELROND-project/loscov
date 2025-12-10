import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import *


def get_correlations(cl_intp, Thetamin, Thetamax, nTheta=nTheta):
    """
    Generates the + and - correlation functions from the angular power spectrum
    using Hankel transformations, for a range of apertures [Thetamin, Thetamax]
    expressed in radians. This works for the shear and shape correlations.
    It returns the correlation functions as arrays.

    cl_intp - the interpolated cls
    Thetamin - minimum value of theta to interpolate from (in rad)
    Thetamax - maximum value of theta to interpolate to (in rad)
    nTheta - the number of points in the correlation
    """
    
    # initialise Hankel transforms
    ht0 = HankelTransform(nu=0, N=1e4, h=1e-2) # for xi_plus
    ht4 = HankelTransform(nu=4, N=1e4, h=1e-2) # for xi_minus
    
    # define angular domain
    logThetamin = np.log10(Thetamin)
    logThetamax = np.log10(Thetamax)
    Theta = np.logspace(logThetamin, logThetamax, nTheta)
    Theta = np.append([0], Theta)
    
    xi_plus = ht0.transform(cl_intp, Theta, ret_err=False)/(2*np.pi)
    xi_minus = ht4.transform(cl_intp, Theta, ret_err=False)/(2*np.pi)
    
    xi_p = (1/2) * (xi_plus + xi_minus)
    xi_x = (1/2) * (xi_plus - xi_minus)
    
    return Theta, xi_p, xi_x, xi_plus, xi_minus

def get_DD_correlations(cl_DD_intp, Thetamin, Thetamax, nTheta=nTheta):
	"""
	A function that computes the integrated correlation function 
    for delta - delta in flat sky, using Hankel transform for a 
    range of apertures [Thetamin, Thetamax] expressed in radians. 
    It returns the correlation functions as arrays.

    cl_intp - the interpolated cls
    Thetamin - minimum value of theta to interpolate from (in rad)
    Thetamax - maximum value of theta to interpolate to (in rad)
    nTheta - the number of points in the correlation
	"""
		
	# initialise Hankel transforms
	ht0 = HankelTransform(nu=0, N=1e4, h=1e-2) 
		
	# define angular domain
	Theta = np.linspace(Thetamin, Thetamax, nTheta)
		
	xi = ht0.transform(cl_DD_intp, Theta, ret_err=False)/(2*np.pi)
		
	return Theta, xi

def get_gD_correlations(cl_gD_intp, Thetamin, Thetamax, nTheta=nTheta):
	"""
	Generates the LOS shear (or cosmic shear)-Delta correlation 
    function, using Hankel transformations, for a range of apertures 
    [Thetamin, Thetamax] expressed in radians. 
    It returns the correlation functions as arrays.

    cl_intp - the interpolated cls
    Thetamin - minimum value of theta to interpolate from (in rad)
    Thetamax - maximum value of theta to interpolate to (in rad)
    nTheta - the number of points in the correlation
	"""
		
	# initialise Hankel transforms
	ht2 = HankelTransform(nu=2, N=1e4, h=1e-2) 
	 
	# define angular domain
	Theta = np.linspace(Thetamin, Thetamax, nTheta)
		
	xi_gD = ht2.transform(cl_gD_intp, Theta, ret_err=False)/(2*np.pi)
		
	return Theta, xi_gD
