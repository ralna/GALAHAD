LHS
===

.. module:: galahad.lhs

The ``lhs`` package **computes an array of Latin Hypercube samples**.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/lhs.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: lhs.initialize()

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
             * < 1 no output
             * > 0 debugging.
          duplication : int
             the duplication factor. This must be at least 1, a value
             of 5 is reasonable.
          space_critical : bool
             if %space_critical True, every effort will be made to use
             as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if %deallocate_error_fatal is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] lhs.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             return status. See LHS_solve for details.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.

   .. function:: lhs.finalize()

     Deallocate all internal private storage.
