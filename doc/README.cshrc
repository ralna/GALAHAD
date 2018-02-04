## If you are using the C shell (csh) or the enhanced C shell (tcsh),
## what you need to add to your .cshrc file looks similar to the following:

## for precise locations, see the output from the install script

## -----------------------
## SIFDecode (if required)
## -----------------------

## This sets the environment variable SIFDECODE to the main SIFDecode directory
## NB. This must correspond to the output from the install script

setenv SIFDECODE ${HOME}/sifdecode

## This updates the search path for manual pages

setenv MANPATH ${SIFDECODE}/man:${MANPATH}

## This updates the search path for executables

set path=(${SIFDECODE}/bin $path)

## --------------------
## CUTEst (if required)
## --------------------

## This sets the environment variable CUTEST to the main CUTEst directory
## NB. This must correspond to the output from the install script

setenv CUTEST ${HOME}/cutest

## This updates the search path for manual pages

setenv MANPATH ${CUTEST}/man:${MANPATH}

## This updates the search path for executables

set path=(${CUTEST}/bin $path)

## This updates the environment variable MATLABPATH

setenv MATLABPATH ${CUTEST}/src/matlab:$(MATLABPATH)

## ---------------------
## GALAHAD (if required)
## ---------------------

## This sets the environment variable GALAHAD to the main GALAHAD directory
## NB. This must correspond to the output from the install script

setenv GALAHAD ${HOME}/galahad

## This updates the search path for manual pages

setenv MANPATH ${GALAHAD}/man:${MANPATH}

## This updates the search path for executables

set path=(${GALAHAD}/bin $path)

## This updates the environment variable MATLABPATH

setenv MATLABPATH ${GALAHAD}/src/matlab:$(MATLABPATH)

## -------------------------------
## SIF test problems (if required)
## -------------------------------

## This sets the environment variable MASTSIF pointing to your source of 
## SIF problems

setenv MASTSIF ${SIFDECODE}/sif

