import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *

from functions.useful_functions import *

def optimise_bins(correlation_function, correlation_type, antiderivative, b, SNR_goal = 20, Nbin_max = 10, SNR_min = 1.5, Thetamax_distribution = Thetamax_dist):

    Thetamax_distribution = arcmintorad(Thetamax_distribution)
    
    get_item('L0', 'E0') #we'll need these for the prefactors in the signal to noise ratios
    
    redshift_distributions = load_file(f"data/{suffix}/redshift_distributions")

    if correlation_type == 'LE':
        G_B = redshift_distributions['E'].get_ngal(b)

    elif correlation_type == 'LP':
        G_B = redshift_distributions['P'].get_ngal(b)
    
    def snr(theta_1, theta_2):

        numerator = antiderivative(theta_2) - antiderivative(theta_1)
        denom_sq = theta_2**2 - theta_1**2
        
        if denom_sq <= 0:
            return 0

        denominator = np.sqrt(denom_sq)
        
        if correlation_type == 'LL':
            A = 2 * np.sqrt(np.pi / Omegatot) * Nlens / (sigma_L**2 + L0)
        
        elif correlation_type == 'LE': #check the factors of 2
            A = 2 * np.sqrt(np.pi / Omegatot) * np.sqrt( 2 * Nlens * G_B / ( (sigma_L**2 + L0) * (sigma_E**2 + E0[b]) ) )
        
        elif correlation_type == 'LP': #check the factors of 2
            A = 2 * np.sqrt(np.pi / Omegatot) * np.sqrt( 2 * Nlens * G_B / (sigma_L**2 + L0) )
        
        return A * numerator / denominator

    def SNR(theta):
        return snr(0,theta)
        
    theta_optimal, SNR_max = find_maximum(SNR, 0, Thetamax_distribution)

    if SNR_max < SNR_goal:               #if our goal SNR is unachievable, even with just 1 bin
        return [0, theta_optimal]         #return a single bin (the best we can do)

    #if the above isn't satisfied, then it means that SNR_goal < SNR_max. This means that 
    
    else: #if our goal SNR is in fact achievable, we want to iteratively determine our bins
        
        #the SNR we use is either our goal SNR or the SNR limited by the max number of bins we want
        SNR_use = max(SNR_goal, SNR_max / ( np.sqrt( 0.5 * Nbin_max ) ) )    #NB this is not exact, and purely empirical

        binscheme = [0] #our bins start at zero    

        alpha = 0
        
        finished = False
        
        while not finished:
            
            def SNR_to_optimise(theta):
                """
                A function which goes to zero when the integrated snr between the bin limit on the left and some theta value is equal to the SNR we want to use. 
                To get to this point, we must have already passed the test that SNR_goal < SNR_max. Therefore, both SNR_goal and SNR_max / Nbin_max exist within
                the range [ binscheme[alpha], theta_optimal ], and therefore so too must their maximum, so this function will always have an x-intercept.

                Note that, for this method to work, the snr should increase monotonically up to the maximum, but it seems that this is indeed the case (check for issues!)
                """
                
                return snr(binscheme[alpha], theta) - SNR_use
   
            theta_new = root_scalar(SNR_to_optimise, bracket=[binscheme[alpha], theta_optimal], method='brentq', xtol=1e-8).root #this solves for the value of theta which gets us to our desired snr

            alpha += 1
            binscheme.append(theta_new)

            #once we've found the new bin limit, we must once again define a function which returns the SNR from that bin limit to some arbitrary theta  
            def SNR(theta):
                return snr(binscheme[alpha],theta)

            #we then want to find what the maximum SNR beyond this point is
            theta_optimal, SNR_max = find_maximum(SNR, theta_new, Thetamax_distribution)

            #if our goal SNR is unachievable with the remaining binning 
            if SNR_max < SNR_use:     

                #if, however, the SNR is still acceptable,
                if SNR_max >= SNR_min:

                    #the optimal theta becomes the final bin limit
                    binscheme.append(theta_optimal)

                #if the SNR in the last bin would be unusable,
                else:

                    #we need a function which returns the SNR from the previous bin limit (before the most recently added one) to some arbitrary theta
                    def SNR(theta):
                        return snr(binscheme[-2],theta)

                    #this maximum might occur at the most recent bin limit, or at a higher value of theta
                    theta_optimal_new, SNR_max_new = find_maximum(SNR, binscheme[-2], Thetamax_distribution)

                    #either way, the theta we obtain in the above becomes our new final bin limit
                    binscheme[-1] = theta_optimal_new

                #either way, provided SNR_max < SNR_use, our while loop is done
                finished = True

        return binscheme        


class Angular_Distributions:
    """
    This class produces anything useful related to the statistics of lenses and galaxies,
    and their binning in angular separation
    """
    
    def __init__(self, Nobjects, binscheme=None, sky_coverage=sky_coverage, Nbin_a=None, Thetamax=Thetamax_dist):
        """
        Arguments:
        - Nlens         : number of lenses we can use
        - Ngal          : number of galaxies we can use
        - sky_coverage  : area of the survey footprint, in deg2 
        - Nbin_a         : number of bins of angular separation
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
            
            Omega_arcmin2 = np.pi * Thetamax**2 / Nbin_a  #the angular size of each bin
            self.Nbina = Nbin_a
            self.Omega = Omega_arcmin2 * (np.pi / 180 / 60)**2  # converting the angular size of a bin to rad2
            self.Omegas = self.Omega * np.ones(Nbin_a) # in rad2 
            
            # the list of Omegas is in case we want to have different bin sizes
            self.limits = np.sqrt(self.Omega / np.pi * np.arange(Nbin_a + 1)) # in rad        
            
            # "Centres" of the bins, taken as the mean separation of objects within the angular bin
            self.Thetas = 2/3 * (self.limits[1:]**3  - self.limits[:-1]**3 ) / (self.limits[1:]**2  - self.limits[:-1]**2 ) 
            
        elif isinstance(binscheme, list):
                
            Omegas = []  #the angular size of each bin, in rad^2
            limits = []
            limits.append(binscheme[0])    #the minimum angular separation considered
                
            Nbin_a = len(binscheme) - 1
            self.Nbina = Nbin_a 
                
            for i in range(Nbin_a):
                omega = np.pi * (binscheme[i+1]**2-binscheme[i]**2) #the angular size of that bin, pi x (R_outer^2 - r_inner^2) 
                Omegas.append(omega)
                limits.append(binscheme[i+1])
                
            self.Omegas = np.array(Omegas)
                
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


def generate_binned_correlation(correlation, b1):
    """Handles generation of binned correlation functions based on cov_matrix type."""
    correlation_data = {}

    get_item('LL_plus','LL_minus')
    get_item('LE_plus','LE_minus')
    get_item('LP')
    get_item('angular_distributions')

    correlation_mapping = {
        "LL": lambda: {
            "plus_correlation": angular_distributions['LL_plus'].compute_binned_correlation(LL_plus),
            "minus_correlation": angular_distributions['LL_minus'].compute_binned_correlation(LL_minus),
        },
        "LE": lambda: {
            "plus_correlation": angular_distributions['LE_plus'][b1].compute_binned_correlation(LE_plus[b1]),
            "minus_correlation": angular_distributions['LE_minus'][b1].compute_binned_correlation(LE_minus[b1]),
        },
        "LP": lambda: {
            "correlation": angular_distributions['LP'][b1].compute_binned_correlation(LP[b1]),
        }
    }

    if correlation in correlation_mapping:
        correlation_data = correlation_mapping[correlation]()
    
    return correlation_data
