Matlab interfaces are now available to a growing number of GALAHAD packages.

                            ---------
                            For LINUX
                            ---------

                       ....................
                       MATLAB before R2011a
                       ....................

To use the Matlab interfaces, GALAHAD must be installed using the
g95 compiler - other compilers may become available in the future
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

                     ........................
                     MATLAB for R2011a-R2013a
                    .........................

For MATLAB R2011a and above, GALAHAD must be installed using the
gfortran compiler. BUT ... you will need to use (and have installed)
gcc/gfortran-4.4, not the more modern version (4.5 and above) that comes
as default with today's Linux. For Ubuntu Linux, see

   https://help.ubuntu.com/community/MATLAB

for details on how to download packaged versions of the relevant
outdated compilers. For other Linux distributions, you might have
to build gcc-4.4 from source code. Grumble to the Mathworks!

Once you have a working gcc/gfortran 4.4 - making sure that they are
both on your shell search path - either select the "install with Matlab"
option when installing, or if a gfortran version has already been installed,
issue the commands

  cd $GALAHAD/src/matlab
  make -s -f $GALAHAD/makefiles/pc.lnx.gfo

(substitute pc.lnx.gfo for the appropriate string on non-Linux or 64-bit
 machines).

N.B. the MATLAB environment variable must point to your system matlab directory.

Once the Matlab versions have been installed, make sure that
$GALAHAD/src/matlab is on your Matlab path.

 Issue the commands

  help galahad

to find the current status of available interfaces.

                     ........................
                     MATLAB for R2013b-R2016a
                    .........................

As for MATLAB for R2011a-R2013a, but you will need gfortran/gcc-4.7 not 4.4.
Edit $GALAHAD/makefiles/pc.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -4.7.

                     ...........................
                     MATLAB for R2016b and above
                    ...........................

As for MATLAB for R2011a-R2013a, but you will need gfortran/gcc-4.9 not 4.4.
Edit $GALAHAD/makefiles/pc.lnx.gfo to check that all mentions of gfortran/gcc
have the trailing -4.9. This should have been taken care of automatically
during the installation process

                         ------------
                         For MAC OS X
                         ------------

Here, the supported compiler is GNU gfortran. So

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

Nick Gould          (nick.gould@stfc.ac.uk)
Dominique Orban     (dominique.orban@polymtl.ca)
Philippe Toint      (philippe.toint@fundp.ac.be)


 ---------------------------------------------------------------------------
 The dreaded "MATLAB Error: Cannot load any more object with static TLS" bug
 ---------------------------------------------------------------------------

For a number of years, the Mathworks have been aware of a serious mex bug.
MATLAB dynamically loads some libraries with static TLS (thread local storage,
e.g. see gcc compiler flag -ftls-model). Loading too many such libs eventually
leaaves no space left. The Mathworks solution is a farcical work-around
that involves inserting the line "ones(10)*ones(10);" in startup.m, that
aims to load "important" libraries first. Of late, this "solution" seems
increasigly less likley to work, but Mathworks appears not to care about
its significant mex user base by failing to provide a proper fix.

The most successful temporary "cure" we have found is to compile the
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

For GALAHAD productions
This version: 1st March 2017

