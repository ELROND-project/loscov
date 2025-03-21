import numpy as np
import config
from functions import useful_functions as uf 


class Distributions:
    """
    This class produces anything useful related to the statistics of lenses and galaxies,
    and their binning in angular separation
    """
    
    def __init__(self, Nobjects, binscheme, sky_coverage, Nbina, Thetamax):

        """
        Arguments:
        - Nlens         : number of lenses we can use
        - Ngal          : number of galaxies we can use
        - sky_coverage  : area of the survey footprint, in deg2 
        - Nbina         : number of bins of angular separation
        - b1            : the first angular separation bin
        - b2            : the second angular separation bin
        - Thetamax      : maximum angular separation, in arcmin
        - binscheme     : list, min and max angular separations of each angular separation bin
        
        All the angular attributes will be expressed in rad.
        """
        
        # Lens number and density
        self.Omegatot = sky_coverage * (np.pi / 180)**2 # in rad2
        self.number = Nobjects
        self.density = Nobjects / self.Omegatot #the density of lenses or galaxies in an angular bin

        # Binning
        if isinstance(binscheme, int): 
            
            Omega_arcmin2 = np.pi * Thetamax**2 / Nbina  #the angular size of each bin
            self.Nbina = Nbina
            self.Omega = Omega_arcmin2 * (np.pi / 180 / 60)**2  # converting the angular size of a bin to rad2
            self.Omegas = self.Omega * np.ones(Nbina) # in rad2 
            
            # the list of Omegas is in case we want to have different bin sizes
            self.limits = np.sqrt(self.Omega / np.pi * np.arange(Nbina + 1)) # in rad        
            
            # "Centres" of the bins, taken as the median separation
            self.Thetas = np.sqrt((self.limits[1:]**2 + self.limits[:-1]**2) / 2) # in rad
            
        elif binscheme is not None:
                
            Omegas = []  #the angular size of each bin, in rad^2
            limits = []
            limits.append(binscheme[0])    #the minimum angular separation considered
                
            Nbina = len(binscheme)
                
            for i in range(Nbina):
                omega = np.pi * (binscheme[i+1]**2-binscheme[i]**2) #the angular size of that bin, pi x (R_outer^2 - r_inner^2) ###NB CHECK UNITS!!!!
                Omegas.append(omega)
                limits.append(binscheme[i+1])
                
            self.Omegas = Omegas # check units!!!! 
                
            # the list of Omegas is in case we want to have different bin sizes
            self.limits = np.array(limits)        
                
            # "Centres" of the bins, taken as the median separation
            self.Thetas = np.sqrt((self.limits[1:]**2 + self.limits[:-1]**2) / 2) # in rad
            
        else:
            print("Error: no binscheme defined")        

    
    def compute_binned_correlation(self, correlation):
        """
        This function computes the expected signal for a correlation function with this binning
        Note, this is only relevant for autocorrelation functions.
        """
        
        xi_bins  = []
        
        Nbin   = self.Nbina
        rs     = self.limits
        Omegas = self.Omegas

        def integrand(r):
            f = 2 * np.pi * r * correlation(r)
            return f
        
        for a in range(Nbin):

            integral, err = monte_carlo_integrate(integrand, [(rs[a], rs[a+1])])
            integral /= Omegas[a]
            xi_bins.append(integral)

        return xi_bins


def generate_binned_correlation(distributions, cov_matrix, b1, b2):
    """Handles generation of binned correlation functions based on cov_matrix type."""
    correlation_data = {}

    uf.get_item('xi2_LOS_plus_intp','xi2_LOS_minus_intp')
    uf.get_item('xi2_LOS_eps_plus_intp','xi2_LOS_eps_minus_intp')
    uf.get_item('xi2_LOS_d_intp')

    correlation_mapping = {
        "LLLL": lambda: {
            "plus_correlation": distributions['LL'].compute_binned_correlation(xi2_LOS_plus_intp),
            "minus_correlation": distributions['LL'].compute_binned_correlation(xi2_LOS_minus_intp),
        },
        "LeLe": lambda: {
            "plus_correlation": distributions['Le'].compute_binned_correlation(xi2_LOS_eps_minus_intp[b1]),
            "minus_correlation": distributions['Le'].compute_binned_correlation(xi2_LOS_eps_minus_intp[b1]),
        } if b1 == b2 else None,
        "LpLp": lambda: {
            "correlation": distributions['Lp'].compute_binned_correlation(xi2_LOS_d_intp[b1]),
        } if b1 == b2 else None
    }

    if cov_matrix in correlation_mapping:
        correlation_data = correlation_mapping[cov_matrix]()
    
    return correlation_data
