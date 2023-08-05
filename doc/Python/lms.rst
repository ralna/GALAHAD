LMS
===

.. module:: galahad.lms

.. include:: lms_intro.rst

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

             * **1** 

               BFGS (the default).

             * **2** 

               Symmetric Rank-One (SR1).

             * **3** 

               The inverse of the BFGS formula.

             * **4** 

               The inverse of the shifted BFGS formula. This should
               be used instead of ``method`` = 3 whenever a shift is
               planned.
          any_method : bool
             allow space to permit different methods if required (less
             efficient).
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

   .. function:: [optional] lms.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             the return status.  Possible values are:

             * **0**

               The call was successful.

             * **-1**

               An allocation error occurred. A message indicating the
               offending array is written on unit options['error'], and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-2**

               A deallocation error occurred.  A message indicating the
               offending array is written on unit options['error'] and
               the returned allocation status and a string containing
               the name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-3**

               One of the restrictions n > 0, delta > 0, lambda > 0 or
               $s^T y > 0$ has been violated and the update has
               been skipped.

             * **-10**

               The matrix cannot be built from the current vectors 
               $\{s_k\}$ and $\{y_k\}$ and values  $\delta_k$ and $\lambda_k$ 
               and the update has been skipped.

             * **-31**

               A call to the function ``lhs_apply`` has been made without 
               a prior call to ``lhs_form_shift`` or ``lhs_form`` with lambda 
               specified  when options['method'] = 4, or ``lhs_form_shift`` 
               has been called when  options['method'] = 3, or 
               ``lhs_change_method`` has been called after
               options['any_method'] = False was specified when calling 
               ``lhs_setup``.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
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
