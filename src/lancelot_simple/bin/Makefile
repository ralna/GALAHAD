########################################################################
#                                                                      #
# Makefile                                                             #
#                                                                      #
#    compiles and links the simple 1-element naive interface           #
#    to LANCELOT B, with the program RUN_LANCELOT_simple.f90.          #
#    This makefile requires the prior installation of the              #
#    GALAHAD package, including the definition of the GALAHAD          #
#    environment variable (This variable needs to point to the main    #
#    directory in which the GALAHAD package has been installed.)       #
#                                                                      #
#    D. Orban, December 2007.                                          #
#    Copyright reserved, Gould/Orban/Toint, for GALAHAD productions    #
#                                                                      #
########################################################################

FORTRAN = lf95
ARCH    = pc.lnx.l95
PREC    = double

MAIN    = run_lancelot_simple
USRLIBS =

GALMODULES = ${GALAHAD}/modules/$(ARCH)/$(PREC)
GALLIBDIR  = ${GALAHAD}/objects/$(ARCH)/$(PREC)
GALLIBS1   = -lgalahad -lgalahad_hsl -lgalahad_lapack
GALLIBS2   = -lgalahad_metis -lgalahad_blas 
GALLIBS    = -L$(GALLIBDIR) $(GALLIBS1) $(GALLIBS2)

all:: $(MAIN)

%.o: %.f90
	$(FORTRAN) -c -I$(GALMODULES) $+

$(MAIN): $(MAIN).o
	$(FORTRAN) -o $@ $+ $(GALLIBS) $(USRLIBS) 

clean::
	rm -f $(MAIN) *.o *.mod  

