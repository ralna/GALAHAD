SEC
===

.. module:: galahad.sec

.. include:: sec_intro.rst

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
