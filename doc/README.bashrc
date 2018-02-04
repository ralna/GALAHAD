## If you are using the bourne again shell, (bash) what you need to add to 
## your .bashrc file looks similar to the following.

## for precise locations, see the output from the install script

## -----------------------
## SIFDecode (if required)
## -----------------------

## This sets the environment variable SIFDECODE to the main SIFDecode directory
## NB. This must correspond to the output from the install script

export SIFDECODE="${HOME}/sifdecode"

## This updates the search path for manual pages

export MANPATH="${SIFDECODE}/man:${MANPATH}"

## This updates the search path for executables

export PATH="${SIFDECODE}/bin:$PATH"

## --------------------
## CUTEst (if required)
## --------------------

## This sets the environment variable CUTEST to the main CUTEst directory
## NB. This must correspond to the output from the install script

export CUTEST="${HOME}/cutest"

## This updates the search path for manual pages

export MANPATH="${CUTEST}/man:${MANPATH}"

## This updates the search path for executables

export PATH="${CUTEST}/bin:$PATH"

## This updates the environment variable MATLABPATH

export MATLABPATH="${CUTEST}/src/matlab:${MATLABPATH}"

## ---------------------
## GALAHAD (if required)
## ---------------------

## This sets the environment variable GALAHAD to the main GALAHAD directory
## NB. This must correspond to the output from the install script

export GALAHAD="${HOME}/galahad"

## This updates the search path for manual pages

export MANPATH="${GALAHAD}/man:${MANPATH}"

## This updates the search path for executables

export PATH="${GALAHAD}/bin:$PATH"

## This updates the environment variable MATLABPATH

export MATLABPATH="${GALAHAD}/src/matlab:${MATLABPATH}"

## -------------------------------
## SIF test problems (if required)
## -------------------------------

## This sets the environment variable MASTSIF pointing to your source of 
## SIF problems

export MASTSIF="${SIFDECODE}/sif"

