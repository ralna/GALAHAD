ROOTS
=====

.. module:: galahad.roots

The ``roots`` package uses classical formulae together with Newton’s method 
to **find all the real roots of a real polynomial.**

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/roots.pdf for additional details.

method
------

Littlewood and Ferrari's algorithms are used to find estimates of the
real roots of cubic and quartic polynomials, respectively; a stabilized version 
of the well-known formula is used in the quadratic case. Newton's
method and/or methods based on the companion matrix are used to further 
refine the computed roots if necessary. Madsen and Reid's (1975) 
method is used for polynomials whose degree exceeds four.

reference
---------

The basic method is that given by

  K. Madsen and J. K. Reid, 
  ``FORTRAN Subroutines for Finding Polynomial Zeros''.
  Technical Report A.E.R.E. R.7986, Computer Science and System Division, 
  A.E.R.E. Harwell, Oxfordshire (1975)

functions
---------

   .. function:: roots.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required is specified by print_level.
          tol : float
             the required accuracy of the roots.
          zero_coef : float
             any coefficient smaller in absolute value than zero_coef
             will be regarde to be zero.
          zero_f : float
             any value of the polynomial smaller in absolute value than
             zero_f will be regarded as giving a root.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] roots.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             the return status.  Possible values are:

             * **0**

               The call was succesful.

             * **-1**

               An allocation error occurred. A message indicating the
               offending array is written on unit control['error'], and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-2**

               A deallocation error occurred.  A message indicating the
               offending array is written on unit control['error'] and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-3**

               Either the specified degree of the polynomial in ``degree`` 
               is less than 0, or the declared dimension of the array ``roots`` 
               is smaller than the specified degree.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.

   .. function:: roots.finalize()

     Deallocate all internal private storage.