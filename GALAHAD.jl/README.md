Julia and GALAHAD for idiots (i.e., me)
=======================================

Prepare GALAHAD binary shared libraries using Meson
---------------------------------------------------

See $GALAHAD/README.meson

but use --prefix=/usr/local in setup to ensure
that the binaries are put in /usr/local/lib

Set the directory containing the galahad shared libraries
---------------------------------------------------------

In your current shell

setenv JULIA_GALAHAD_LIBRARY_PATH /usr/local/lib
or
export JULIA_GALAHAD_LIBRARY_PATH=/usr/local/lib

Setup julia to run GALAHAD
--------------------------

(see $GALAHAD/GALAHAD.jl/README.md for additional details)

cd $GALAHAD/GALAHAD.jl

julia

] key to enter package mode
dev .
backspace key to enter julia mode

ENV["JULIA_GALAHAD_LIBRARY_PATH"] = "/usr/local/lib"
force_recompile(package_name::String) =  
Base.compilecache(Base.identify_package(package_name))
force_recompile("GALAHAD")
import Pkg; Pkg.add("Accessors")
import Pkg; Pkg.add("Quadmath")

Run GALAHAD in julia
--------------------

to test the package snls with julia

include("test/test_snls.jl")

or for all tests

include("test/runtests.jl")

and to get out of julia

exit()

Nick Gould
