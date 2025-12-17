import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from config import *


def redshift_distribution_Euclid(z):
    """
    This is the total redshift distribution of galaxies
    expected from the Euclid survey.
    Source: eq. (10) of https://arxiv.org/pdf/2010.07376.pdf
    """
    
    a = 0.4710
    b = 5.1843
    c = 0.7259
    A = 1.75564 # ensures normalisation to 1
    
    n = A * (z**a + z**(a*b)) / (z**b + c)
    
    return n
    
class Redshift_Distributions:
    """
    This class produces anything useful related to the redshift binning of galaxies
    """
    
    def __init__(self, Nobjects, binscheme, Nbinz = Nbin_z, zmax_dist = zmax_dist, starting_distribution = redshift_distribution_Euclid):
        """
        Arguments:
        - Nobjects      : number of objects we can use
        - binscheme     : list, min and max redshifts of each redshift bin (if None, bins are equally populated)
        - Nbinz         : number of bins of redshift
        - zmax_dist     : maximum redshift (expect 3)
        
        All the angular attributes will be expressed in rad.
        """

        self.Nobjects = Nobjects
        self.Nbinz = Nbinz
        self.zmax_dist = zmax_dist
        
        self.starting_distribution = starting_distribution
        self.binscheme = binscheme

        total, _ = quad(starting_distribution, 0, self.zmax_dist)    #the total value of the distribution (needed for the normalisation)
        self.norm_factor = total

        # Binning
        if isinstance(binscheme, int) or binscheme is None:
    
            # Find redshift values that divide the CDF into Nbinz equal parts
            
            bin_edges = [0.0]   #initialise the list of bin edges with 0
            
            for i in range(1, Nbinz):    
                target = i / Nbinz      #we want the target cumulative distribution to be i / Nbinz of the total  
                sol = root_scalar(lambda z: self.cdf(z) - target, bracket=[0, self.zmax_dist], method='brentq')  #looking for the solution z to the equation cdf(z) - target = 0, within [0, zmax]
                bin_edges.append(sol.root) #whatever that solution is gets appended as the new bin edge
                
            bin_edges.append(zmax_dist) #we then close the last bin
    
            self.limits = np.array(bin_edges)

        elif isinstance(binscheme, (list, np.ndarray)):
            self.limits = np.array(binscheme)  
            
        else:
            raise ValueError("Invalid binscheme: must be None, int, or array-like.")      


    def overall_distribution(self, z):

        return self.starting_distribution(z) / self.norm_factor
    
    def cdf(self, z):
        """
        A cumulative distribution function
        """
        
        result, _ = quad(self.overall_distribution, 0, z)
        
        return result
        
    def pb(self, z, b):
        """
        This is the redshift distribution of galaxies for each redshift bin, normalised for that bin.
    
        z  : the redshift
        b  : the redshift bin (0 to Nbin_z)
        """
        
        zzmin = self.limits[b]
        zzmax = self.limits[b + 1]

        if zzmin <= z < zzmax:
            normalisation_factor, _ = quad(self.overall_distribution, zzmin, zzmax) #the integral of the distribution in that bin
            distrib = self.overall_distribution(z) / normalisation_factor #renormalising for that bin
            
        else:
            distrib = 0
        
        return distrib

    
    def find_bin(self, z):
        """
        This function finds the redshift bin to which a galaxy at redshift z would belong
        """
        for i in range(self.Nbinz):
            if self.limits[i] <= z < self.limits[i + 1]:
                return i
        print('Warning: redshift out of bin range')
        return None

    def get_ngal(self, b):
            """
            Returns the number of galaxies in redshift bin `b`.
    
            b : index of the redshift bin (0 to Nbinz)
            """
            zzmin = self.limits[b]
            zzmax = self.limits[b + 1]
    
            result, _ = quad(self.overall_distribution, zzmin, zzmax) #the fraction of the overall distribution between zzmin and zzmax
            
            return result * self.Nobjects