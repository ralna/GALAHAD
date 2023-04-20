LMS
===

.. module:: galahad.lms

Given a sequence of vectors
$\{s_k\}$ and $\{y_k\}$ and scale factors $\{\delta_k\}$,
the ``lms`` package 
**obtains the product of a limited-memory secant approximation** 
$H_k$ (or its inverse) with a given vector,
using one of a variety of well-established formulae.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/lms.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: lms.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             controls level of diagnostic output.
          memory_length : int
             limited memory length.
          method : int
             limited-memory formula required (others may be added in
             due course):
             * 1 BFGS (default)
             * 2 SR1
             * 3 The inverse of the BFGS formula
             * 4 The inverse of the shifted BFGS formula. This should
             be used  instead of ``method`` = 3 whenever a shift is
             planned.
          any_method : bool
             allow space to permit different methods if required (less
             efficient).
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] lms.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             return status. See LMS_setup for details.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          length : int
             the number of pairs (s,y) currently used to represent the
             limited-memory matrix.
          updates_skipped : bool
             have (s,y) pairs been skipped when forming the
             limited-memory matrix.
          time : dict
             dictionary containing timing information:
               total : float
                  total cpu time spent in the package.
               setup : float
                  cpu time spent setting up space for the secant
                  approximation.
               form : float
                  cpu time spent updating the secant approximation.
               apply : float
                  cpu time spent applying the secant approximation.
               clock_total : float
                  total clock time spent in the package.
               clock_setup : float
                  clock time spent setting up space for the secant
                  approximation.
               clock_form : float
                  clock time spent updating the secant approximation.
               clock_apply : float
                  clock time spent applying the secant approximation.

   .. function:: lms.finalize()

     Deallocate all internal private storage.
