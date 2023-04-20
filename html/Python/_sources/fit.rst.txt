FIT
===

.. module:: galahad.fit

The ``fit`` package **fits polynomials to function and derivative data$$.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/fit.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: fit.initialize()

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
          space_critical : bool
             if space_critical is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation times.
          deallocate_error_fatal : bool
             if deallocate_error_fatal is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] fit.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             return status. Possible values are:
             * 0 Normal termination with the required fit
             * -1 An allocation error occured; the status is given in
             the component  ``alloc_status``
             * -2 A deallocation error occured; the status is given in
             the  component alloc_status
             * - 3 the restriction n >= 1 has been violated.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.

   .. function:: fit.finalize()

     Deallocate all internal private storage.
