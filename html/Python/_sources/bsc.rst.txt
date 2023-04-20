BSC
===

.. module:: galahad.bsc

The ``bsc`` package takes given matrices $A$ and (diagonal) $D$, and
**builds the Schur complement** $S = A D A^T$</b> in sparse co-ordinate 
(and optionally sparse column) format(s). Full advantage is taken 
of any zero coefficients in the matrix $A$.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.

See Section 4 of $GALAHAD/doc/bsc.pdf for a brief description of the
method employed and other details.

functions
---------

   .. function:: bsc.initialize()

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
          max_col : int
             maximum permitted number of nonzeros in a column of
             \f$A\f$; -ve means unlimit.
          new_a : int
             how much has $A$ changed since it was last accessed:
             * 0 = not changed,
             * 1 = values changed,
             * 2 = structure changed
             * 3 = structure changed but values not required.
          extra_space_s : int
             how much extra space is to be allocated in $S$ above that
             needed to hold the Schur complement.
          s_also_by_column : bool
             should s.ptr also be set to indicate the first entry in
             each column of $S$.
          space_critical : bool
             if ``space_critical`` True, every effort will be made to
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

   .. function:: [optional] bsc.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             return status. See BSC_form for details.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          max_col_a : int
             the maximum number of entries in a column of $A$.
          exceeds_max_col : int
             the number of columns of $A$ that have more than
             control.max_col entries.
          time : float
             the total CPU time spent in the package.
          clock_time : float
             the total clock time spent in the package.

   .. function:: bsc.finalize()

     Deallocate all internal private storage.
