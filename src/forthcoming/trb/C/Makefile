FC = gfortran -fbounds-check -Wall -g
CC = gcc-11 -fbounds-check -Wall -g

GINC = -I/Users/jari/Code/GALAHAD/modules/mac64.osx.gfo/double
GLIBS = -L/Users/jari/Code/GALAHAD/objects/mac64.osx.gfo/double \
-lgalahad -lgalahad_blas -lgalahad_lapack -lgalahad_hsl -lgalahad_spral \
-lgalahad_pardiso -lgalahad_mkl_pardiso -lgalahad_wsmp

all: trbs trbs2 trbs3 trbs4 trbt trbt2

trbs: trbs.o trb_ciface.o trb.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)
trbs2: trbs2.o trb_ciface.o trb.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)
trbs3: trbs3.o trb_ciface.o trb.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)
trbs4: trbs4.o trb_ciface.o trb.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)
trbt: trbt.o trb_ciface.o trb.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)
trbt2: trbt2.o trb_ciface.o trb.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)

%.o: %.c
	$(CC) -c $<
%.o: %.f90
	$(FC) $(GLIBS) $(GINC) -c $<
%.o: ../%.f90
	$(FC) $(GLIBS) $(GINC) -c $<

trb_ciface.o: trb.o

clean:
	-rm *.o *.mod
	-rm trbs
	-rm trbs2
	-rm trbs3
	-rm trbs4	
	-rm trbt
	-rm trbt2
