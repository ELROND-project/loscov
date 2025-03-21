# loscov
Generating correlation functions and covariance matrices for the 6x2pt correlation functions between galaxy positions, galaxy shapes and line-of-sight shears.

## running the code

To run the code, make sure you're in the loscov folder, and then enter

sbatch job.sh

into the command line. This first runs job.sh, which produces the correlation functions if needed, and populates the tasks.txt file with all of the parameter variation for the code. 

For example, we could have "1 2 LeLe scov", which would correspond to b1=1 b2=2 for the redshift bins of the sparsity component of the LeLe covariance. There are 258 of these lines in total, which are then automatically batched with parallel_jobs.sh as 258 tasks with 1 node each. 

The runtime for these tasks are recorded in runtime.log, and the print outputs go into parallel-job-output.log, where you can read if there are any elements of the matrix with uncertainties above the specified tolerance.

The matrices are stored in an output folder (eg matrices_1e6_v1), and you can read them in and plot them with plotter.ipynb.

All the functions which the code needs to run are in the functions folder, where you can find the implementation of the cls, monte carlo integrator, the covariance matrices themselves, etc etc

The most important file when using the code is the config.py file. Here, you can adjust whichever parameters you'd like - the precision of the monte carlo integrator, the number of lenses and galaxies, the binnning, the covariance components to save, the output folder name, etc etc. Basically everything you'd want to adjust should be in there.

## understanding the components of the code

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

