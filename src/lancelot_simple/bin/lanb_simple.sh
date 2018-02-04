#!/bin/bash -f

########################################################################
#                                                                      #
# lanb_simple:                                                         #
#                                                                      #
#    runs the simple 1-element simple interface to LANCELOT B.         #
#    This script uses the program lancelot_simples.f90 and the         #
#    subroutine lancelot_simple.f90 and requires the prior             #
#    installation of the GALAHAD package.                              # 
#                                                                      #
#    Adaptations to the local architecture and installation are        #
#    required in the lines between ====================                #
#                                                                      #
#    NOTE: the executable created is called lancelot_simple and left   #
#          in the execution directory.                                 #
#                                                                      # 
#    Ph. L. Toint, November 2007.                                      #
#                                                                      #
########################################################################
#                                                                      #
#    Copyright reserved, Gould/Orban/Toint, for GALAHAD productions    #
#                                                                      #
########################################################################

# Use: lanb_simple [-h] [-o i] [-l secs]

# where: options -h : print this help and stop execution
#                -o : 0 for silent mode, 1 for brief description of
#                     the stages executed
#                     (Default: -o 0)
#                -l : limit the cputime used to secs seconds
#                     (Default: -l 99999999)

###############################################################################

Initialize() {

    set +C

    #  directory for the main executable file
    EXEC=$PWD

    # OUTPUT = 0 (summary output), = 1 (detailed output from decoder)
    let OUTPUT=0

    # LIMIT = 0 (no cputime limit)
    let LIMIT=0

}

###############################################################################

Display_Usage() {

    echo " Use: ${thisprog} [-h] [-o i] [-l secs]"
    echo ' '
    echo ' where options are one or more of'
    echo '    -h or --help   : print this help and stop execution'
    echo '    -o or --output : verbosity (0 = silent, 1 = brief)'
    echo '                     (Default: -o 0)'
    echo '    -l or --limit  : limits the cputime to secs seconds'
    echo '                     (Default: -l 0, i.e., no limit)'

}

###############################################################################

Parse_Arguments() {

    while [[ $# > 0 ]]; do
        case "$1" in
            -o|--output)   OUTPUT=$2
                           shift
                           ;;
            -l|--limit)    let LIMIT=$2
                           shift
                           ;;
            -h|--help)     Display_Usage
                           exit 0
                           ;;
            *)             printf "Unrecognized option $1\n"
                           ;;
        esac
        shift
    done

}

###############################################################################

thisprog=`basename $0`

if [[ ! $?GALAHAD ]]; then
  echo ' You have not set the GALAHAD environment variable. '
  echo ' This needs to point to the main directory in which '
  echo ' you have installed the GALAHAD package. '
  exit 5
fi

Initialize
Parse_Arguments $@

#==============================================================================
# Set default commands and compiler options.
# Some may be overridden by the later sourcing of $GALAHAD/bin/sys/$1

#  standard unix commands

RM="rm -f"
MV="mv"

#  the command that invokes the FORTRAN 95 compiler and linker

FORTRAN="lf95"

#  directory for double precision GALAHAD modules

GALMODULES=$GALAHAD/modules/pc.lnx.l95/double

#  directory for double precision GALAHAD libraries 

GALLIBS=$GALAHAD/objects/pc.lnx.l95/double

#==============================================================================

# compile the lancelot_1e.f90 program
(( OUTPUT > 0 )) && printf '\nCompiling ...\n'

$FORTRAN -c -I$GALMODULES lancelot_simple.f90 lancelot_simples.f90

# link all the tools files together.
(( OUTPUT > 0 )) && printf '\nLinking all the object files together ...\n'

$FORTRAN -o lancelot_simple lancelot_simple.o lancelot_simples.o -L$GALLIBS -lgalahad -lgalahad_blas -lgalahad_hsl -lgalahad_metis -lgalahad_lapack

[[ $cwd != $EXEC  ]] && $MV lancelot_simple $EXEC/lancelot_simple

# run lancelot_simple on the current problem.
(( OUTPUT > 0 )) && printf '\nRunning lancelot_simple ...\n'

ulimit -t $LIMIT
$EXEC/lancelot_simple

# tidy up the current directory.
$RM $EXEC/lancelot_simple.o $EXEC/lancelot_simples.o
