Matlab interfaces are now available to a growing number of GALAHAD packages
(in double-precision).

                            ---------
                            For LINUX
                            ---------
                    ............................
                     MATLAB for R2022a and later
                    ............................

For MATLAB R2011a and above, GALAHAD must be installed using the
gfortran compiler. BUT ... you will need to use (and have installed)
gcc/gfortran-10, not the more modern version (11 and above) that may 
comes as default with today's Linux. For Ubuntu Linux, see

   https://help.ubuntu.com/community/MATLAB

for details on how to download packaged versions of the relevant
outdated compilers. For other Linux distributions, you might have
to build gcc-10 from source code. Grumble to the Mathworks!

Once you have a working gcc/gfortran-10 - making sure that they are
both on your shell search path - either select the "install with Matlab"
option when installing, or if a gfortran version has already been installed,
issue the commands

  cd $GALAHAD/src/matlab
  make -s -f $GALAHAD/makefiles/pc64.lnx.gfo

(substitute pc.lnx.gfo for the appropriate string on non-Linux or 32-bit
 machines, e.g pc.lnx.gfo).

N.B. the MATLAB environment variable must point to your system matlab directory.

Once the Matlab versions have been installed, make sure that
$GALAHAD/src/matlab is on your Matlab path.

Issue the commands

  help galahad

within Matlab to find the current status of available interfaces.

Note that at present there is no single-precision version.

                    ........................
                     MATLAB for R2020b-2021b
                    ........................

As for MATLAB for R2022a, but you will need gfortran/gcc-8 not 10.
Edit $GALAHAD/makefiles/pc64.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -8. This should have been taken care of automatically
during the installation process.

                    .........................
                     MATLAB for R2018a-R2020a
                    .........................

As for MATLAB for R2022a, but you will need gfortran/gcc-6 not 10.
Edit $GALAHAD/makefiles/pc64.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -6. This should have been taken care of automatically
during the installation process.

                     ........................
                     MATLAB for R2016b-R2017b
                    .........................

As for MATLAB for R2022a, but you will need gfortran/gcc-4.9 not 10
Edit $GALAHAD/makefiles/pc64.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -4.9. This should have been taken care of automatically
during the installation process.

                     ........................
                     MATLAB for R2013b-R2016a
                    .........................

As for MATLAB for R2022a, but you will need gfortran/gcc-4.7 not 10.
Edit $GALAHAD/makefiles/pc64.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -4.7.  This should have been taken care of automatically

                     ........................
                     MATLAB for R2011a-R2013a
                    .........................
As for MATLAB for R2022a, but you will need gfortran/gcc-4.4 not 10.
Edit $GALAHAD/makefiles/pc64.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -4.4.

                       ....................
                       MATLAB before R2011a
                       ....................

To use the Matlab interfaces, GALAHAD must be installed using the
g95 compiler - other compilers have become available since then.
if Matlab's mex facility becomes more fortran-90 friendly. Either
select the "install with Matlab" option when installing, or if
a g95 version has already been installed, issue the commands

  cd $GALAHAD/src/matlab
  make -s -f $GALAHAD/makefiles/pc.lnx.g95

(substitute pc.lnx.g95 for the appropriate string on non-Linux or 64-bit
 machines).

N.B. the MYMATLAB environment variable must point to your system matlab
directory; Matlab's mex executable should be found under $MYMATLAB/bin.

Once the Matlab versions have been installed, make sure that
$GALAHAD/src/matlab is on your Matlab path.

 Issue the commands

  help galahad

to find the current status of available interfaces.

For 64-bit Matlab, make sure that you are using the 64-bit g95 compiler,
and that long integers are used on the compile lines. This requires that
the -i8 and -fPIC flags be added to the BASIC variable in
$GALAHAD/makefiles/pc64.lnx.g95, and to the FFLAGS variable in your Matlab
mexopts.sh file (in the relevant release subdirectory of the .matlab directory
in your home directory).

   https://help.ubuntu.com/community/MATLAB

for details on how to download packaged versions of the relevant
outdated compilers. For other Linux distributions, you might have
to build gcc-4.4 from source code. Grumble to the Mathworks!

Once you have a working gcc/gfortran 4.4 - making sure that they are
both on your shell search path - either select the "install with Matlab"
option when installing, or if a gfortran version has already been installed,
issue the commands

  cd $GALAHAD/src/matlab
  make -s -f $GALAHAD/makefiles/pc64.lnx.gfo

(substitute pc.lnx.gfo for the appropriate string on non-Linux or 32-bit
 machines, e.g pc.lnx.gfo).

N.B. the MATLAB environment variable must point to your system matlab directory.

Once the Matlab versions have been installed, make sure that
$GALAHAD/src/matlab is on your Matlab path.

 Issue the commands

  help galahad

to find the current status of available interfaces.

                    ...........................
                    Replacement BLAS and LAPACK
                    ...........................

You may replace the default BLAS and LAPACK routines by tuned, threaded
versions (e.g., OpenBLAS or MKL). But please be aware that Matlab will ignore
any dynamic/shared (.so) files, so that you will need to use static (.a) 
versions. You can achieve this by editing $GALAHAD/makefiles/pc64.lnx.gfo, 
and setting, for example

 BLAS = -l:libopenblas.a
 LAPACK = -l:libopenblas.a

to enable OpenBLAS (check that you have dowloaded the static version of
OpenBLAS from your favourite package manager, e.g., in the package 
libopenblas-dev, or downloaded OpenBLAS from https://www.openblas.net
and compiled and installed this using make). Note the -l:lib*.a rather
than the dynamic -l* syntax.

                         ------------
                         For MAC OS X
                         ------------

Here, the supported compiler used to be GNU gfortran. So

  cd $GALAHAD/src/matlab
  make -s -f $GALAHAD/makefiles/pc.lnx.gfo

will install the 32-bit version. For the 64-bit version, once again
long integers must be used, and adding -fdefault-integer-8 to the
BASIC and FFLAGS variables as described above will achieve this

Unfortunately, from 2014 Apple no longer supports GNU compilers directly,
and this leaves only a "homebrew" solution. Help needed here, please.

                          -----------
                          For Windows
                          -----------

As we only offer support for GALAHAD in Windows using MinGW/MSYS
GNU environent (see README.windows), and as we have not checked
whether g95/gfortran is compatible with Matlab in this case, we
are sorry but the Windows user is on her/his own. Matlab claim
to support Intel Visual Fortran as their default Windows-fortran
interface.

 ---------------------------------------------------------------------------
 The dreaded "MATLAB Error: Cannot load any more object with static TLS" bug
 ---------------------------------------------------------------------------

For a number of years, the Mathworks were aware of a serious mex bug.
MATLAB dynamically loaded some libraries with static TLS (thread local storage,
e.g. see gcc compiler flag -ftls-model). Loading too many such libs eventually
leaaves no space left. The Mathworks solution was a farcical work-around
that involves inserting the line "ones(10)*ones(10);" in startup.m, that
aims to load "important" libraries first. Of late, this "solution" seems
increasigly less likley to work, but Mathworks appeared not to care about
its significant mex user base by failing to provide a proper fix. Fortunately
sanity has broken out, and you are only likely to see this with older
versions of Matlab.

The most successful temporary "cure" we found was to compile the
Matlab programs as usual, to issue the command

  ldd $GALAHAD/src/matlab/galahad_wcp.mexa64

(replacing mexa64 with your particular mex flavour), and then to

export LD_PRELOAD=list of colon-separated .so files mentioned       (sh/bash)

or

setenv LD_PRELOAD =list of colon-separated .so files mentioned      (csh)

and restart matlab.

For example,

ldd $GALAHAD/src/matlab/galahad_wcp.mexa64
	linux-vdso.so.1 =>  (0x00007fff6ebfe000)
	/usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00002b2ab17ee000)
	/usr/lib/x86_64-linux-gnu/libhwloc.so.5 (0x00002b2ab1b00000)
	/usr/lib/x86_64-linux-gnu/libgomp.so.1 (0x00002b2ab1d40000)
	/usr/lib/x86_64-linux-gnu/libgfortran.so.3 (0x00002b2ab1f6e000)
	/lib/x86_64-linux-gnu/libm.so.6 (0x00002b2ab2295000)
	/lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00002b2ab259b000)
	/lib/x86_64-linux-gnu/libc.so.6 (0x00002b2ab27b2000)
	/usr/lib/x86_64-linux-gnu/libnuma.so.1 (0x00002b2ab2b77000)
	/usr/lib/x86_64-linux-gnu/libltdl.so.7 (0x00002b2ab2d82000)
	/lib/x86_64-linux-gnu/libpthread.so.0 (0x00002b2ab2f8c000)
	/usr/lib/x86_64-linux-gnu/libquadmath.so.0 (0x00002b2ab31aa000)
	/lib/x86_64-linux-gnu/libdl.so.2 (0x00002b2ab33e9000)
	/lib64/ld-linux-x86-64.so.2 (0x00002b2ab1128000)
	libmx.so => not found
	libmex.so => not found

=>

setenv LD_PRELOAD /usr/lib/x86_64-linux-gnu/libstdc++.so.6:/usr/lib/x86_64-linux-gnu/libhwloc.so.5:/usr/lib/x86_64-linux-gnu/libgomp.so.1:/usr/lib/x86_64-linux-gnu/libgfortran.so.3:/lib/x86_64-linux-gnu/libm.so.6:/lib/x86_64-linux-gnu/libgcc_s.so.1:/lib/x86_64-linux-gnu/libc.so.6:/lib64/ld-linux-x86-64.so.2:/usr/lib/x86_64-linux-gnu/libnuma.so.1:/usr/lib/x86_64-linux-gnu/libltdl.so.7:/lib/x86_64-linux-gnu/libpthread.so.0:/usr/lib/x86_64-linux-gnu/libquadmath.so.0:/lib/x86_64-linux-gnu/libdl.so.2

for csh. Many thanks to

http://stackoverflow.com/questions/19268293/matlab-error-cannot-open-with-static-tls

for the discussion and tip

Nick Gould          (nick.gould@stfc.ac.uk)
Dominique Orban     (dominique.orban@polymtl.ca)
Philippe Toint      (philippe.toint@fundp.ac.be)
Jari Fowkes         (jaroslav.fowkes@stfc.ac.uk)
Alexis Montoison    (alexis.montoison@polymtl.ca)

For GALAHAD productions
This version: 24th May 2024
