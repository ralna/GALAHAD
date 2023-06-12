CONVERT
=======

.. module:: galahad.convert

.. include:: convert_intro.rst

functions
---------

   .. function:: convert.initialize()

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
          transpose : bool
             obtain the transpose of the input matrix?.
          sum_duplicates : bool
             add the values of entries in duplicate positions?.
          order : bool
             order row or column data by increasing index?.
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: [optional] convert.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             the return status. Possible values are:

             * **0**

               a successful conversion occurred.

             * **-1** 

               An allocation error occurred. A message indicating
               the  offending array is written on unit control.error, and
               the  returned allocation status and a string containing
               the name  of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-2** 

               A deallocation error occurred. A message indicating
               the  offending array is written on unit control.error and
               the  returned allocation status and a string containing
               the  name of the offending array are held in
               inform['alloc_status'] and inform['bad_alloc'] respectively.

             * **-3**

               The restriction n > 0 or m > 0 or requirement that a
               type  contains its relevant string 'coordinate',
               'sparse_by_rows',  'sparse_by_columns', 'dense_by_rows' or
               'dense_by_columns'  has been violated.

             * **-32**

               provided integer workspace is not large enough.

             * **-33**

               provided real workspace is not large enough.

             * **-73**

               an input matrix entry has been repeated.

             * **-79**

               there are missing optional arguments.

             * **-90**

               a requested output format is not recognised.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          duplicates : int
             the number of duplicates found (-ve = not checked).
          time : dict
             dictionary containing timing information:
                total : float
                   total cpu time spent in the package.
                clock_total : float
                   total clock time spent in the package.

   .. function:: convert.finalize()

     Deallocate all internal private storage.
