# loscov
Generating correlation functions and covariance matrices for the 6x2pt correlation functions between galaxy positions, galaxy shapes and line-of-sight shears.

To run the code, simply enter job.sh in the command line (assuming you're in the loscov folder)

The important bits of the code are as follows:

### config.py

This is the file in which you can adjust (almost) all of the parameters of the run. 

### plotter.ipynb

Here you can plot the covariance matrices, signal vs noise etc

### functions

The folder containing all the functions (correlation functions, covariance matrices etc) needed in the run

### matrices_(number)_(version)

Contains the results of the run for a given max number of samples in the monte carlo integrator and a given version

### output(number)

Contains the outputs and errors of a run (here you can check for large uncertainties, as well as any errors in the runs)

### runtime_(number).log

This tells you how long different pieces of the covariance matrices took to run, with the number of samples in the monte carlo integrator set to (number)

