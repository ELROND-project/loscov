# loscov
Code used in the generation and plotting of correlation functions and covariance matrices for the 6x2pt correlation functions between galaxy positions, galaxy shapes and strong lensing line-of-sight shear, used in Fleury et al. 2026 (link to paper).

This current version is a work in progress, and computes only the covariance of the 3 new observables, LL, LE and LP. It will be updated in future to include the full 6x2pt scheme, as well as with general speed-ups and other improvements.

Authors: Daniel Johnson, Pierre Fleury, Theo Duboscq, Natalie Hogg

## Running the code

To generate the relevant correlation functions and covariance matrices,

1. set run parameters in config.py - here you can specify survey properties, angular and redshift binning schemes, cosmology, numerical parameters etc

2. Run job.sh - this will first compute the galaxy redshift bins, correlation functions, angular distributions and binned correlation functions using correlations_and_distributions.py on a single core. Once this completes, the computation of covariance matrices will be parallelised according to the specifications in config.py.

The resulting data is saved in within the data folder, with a file name specifying key parameters (the repository includes the data for the optimistic and conservative scenarios detailed in Fleury et al. 2026 (link). The covariance matrix, binned correlations and their errors are all plotted using plotter.ipynb. The correlation functions and redshift distributions are plotted with plotting_correlations.ipynb, and the numerical errors from the monte carlo integration are examined in error_analysis.ipynb. 

Note that the .sh files are adapted to the particular computing cluster used by the authors, and should be adapted (as well as the relevant parameters in the config.py file) 

## Attribution

Please cite Fleury et al. 2026 (link) if loscov is used for a publication, and link https://github.com/ELROND-project/loscov. Feel free to get in touch with any questions you might have!
