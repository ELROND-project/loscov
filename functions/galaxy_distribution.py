import numpy as np
from config import binparams

def pb(z, b):
    """
    This is the (normalised) redshift distribution of galaxies expected from the Euclid survey for each redshift bin.
    Source: eq. (10) of https://arxiv.org/pdf/2010.07376.pdf and caption of fig. 3.

    z  : the redshift
    b  : the redshift bin (0 to 4)
    """

    if b in [0,1,2,3,4]:

        A = binparams['A'][b]
        alpha = binparams['alpha']
        beta = binparams['beta']
        gamma = binparams['gamma']
        zzmin = binparams['redshifts'][b] 
        zzmax = binparams['redshifts'][b+1]
        
        if zzmin <= z < zzmax: #if the redshift is in the permitted redshift bin
            distrib = A * (z**alpha + z**(alpha*beta)) / (z**beta + gamma)
        else: 
            distrib = 0
            
        return distrib*np.heaviside(distrib,0)
    
    else:
        raise ValueError('Only bins 0 to 4 are supported.')

def find_bin(z):
    """
    This function finds the redshift bin to which a galaxy at redshift z would belong
    """

    found = False
    
    for i in range(len(binparams['A'])):
        if binparams['redshifts'][i] <= z and z < binparams['redshifts'][i+1]:
            bb = i
            found = True 

    if not found:  #if the galaxy lies outside any bin
        print('bin not found')
    else:
        return bb

def get_ngal(b):
    """
    Calculates the number of galaxies in bin b 
    """
        
    return NGal/5