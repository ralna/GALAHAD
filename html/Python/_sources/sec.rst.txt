SEC
===

.. module:: galahad.sec

The ``sec`` package 
**builds and updates dense BFGS and SR1 secant approximations to a Hessian**
so that the approximation $B$ satisfies the secant condition $B s = y$
for given vectors $s$ and $y$.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/sec.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: sec.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required. Possible values are:

             * **<1** 

               no output will occur.

             * **>0** 

               debugging output will occur.
          h_initial : float
             the initial Hessian approximation will be ``h_initial`` * $I$.
          update_skip_tol : float
             an update is skipped if the resulting matrix would have
             grown too much; specifically it is skipped when
             $y^T s / y^T y \leq$ ``update_skip_tol``.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] sec.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             the return status. Possible valuesa are:

             * **0** 

               successful update occurred.

             * **-85**

               an update is inappropriate and has been skipped.

   .. function:: sec.finalize()

     Deallocate all internal private storage.
