FC = gfortran -fbounds-check -Wall -g
CC = gcc-11 -fbounds-check -Wall -g

GINC = -I/Users/jari/Code/GALAHAD/modules/mac64.osx.gfo/double
GLIBS = -L/Users/jari/Code/GALAHAD/objects/mac64.osx.gfo/double -lgalahad

all: ugo_test ugos

ugo_test: ugo_test.o ugo_ciface.o ugo.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)
ugos: ugos.o ugo_ciface.o ugo.o
	$(FC) -o $@ $^ $(GLIBS) $(GINC)

%.o: %.c
	$(CC) -c $<
%.o: %.f90
	$(FC) $(GLIBS) $(GINC) -c $<
%.o: ../%.f90
	$(FC) $(GLIBS) $(GINC) -c $<

ugo_ciface.o: ugo.o

clean:
	-rm *.o *.mod
	-rm ugo_test
	-rm ugos
