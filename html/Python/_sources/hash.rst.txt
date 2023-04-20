HASH
====

.. module:: galahad.hash

The ``hash`` package **sets up, inserts into, removes from and searches**
a chained scatter table  (Williams, CACM 2, 21-24, 1959).

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/hash.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: hash.initialize()

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
             * <= 0 no output,
             * >= 1 debugging.
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

   .. function:: [optional] hash.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             return status. 0 is a successful call.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.

   .. function:: hash.finalize()

     Deallocate all internal private storage.
