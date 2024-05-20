SHA
===

.. module:: galahad.sha

.. include:: sha_intro.rst

functions
---------

   .. function:: sha.initialize()

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

             * **<=0**

               no output.

             * **1**

               a one-line summary for every iteration.

             * **2**

               a summary of the inner iteration for each iteration.

             * **>=3**

               increasingly verbose (debugging) output.
          approximation_algorithm : int
             which approximation algorithm should be used? Possible values are:

             * **1**

               unsymmetric, parallel (Alg 2.1 in paper).

             * **2**

               symmetric (Alg 2.2 in paper).

             * **3**

               composite, parallel (Alg 2.3 in paper).

             * **4**

               composite, block parallel (Alg 2.4 in paper).

          dense_linear_solver : int
             which dense linear equation solver should be used?
             Possible values are:

             * **1**

               Gaussian elimination.

             * **2**

               QR factorization.

             * **3**

               singular-value decomposition.

             * **4**

               singular-value decomposition with divide-and-conquer.

          extra_differences : int
             if available use an addition extra_differences differences.
          sparse_row : int
             a row is considered sparse if it has no more than .sparse_row
             entries.
          recursion_max : int
             limit on the maximum number of levels of recursion (Alg. 2.4).
          recursion_entries_required : int
              the minimum number of entries in a reduced row that are required
              if a further level of recuresion is allowed (Alg. 2.4).
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to use
             as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.



   .. function:: sha.analyse_matrix(n, ne, row, col, options=None)

      Import problem data into internal storage and compute sparsity-based
      reorderings prior to factorization.

      **Parameters:**

      n : int
          holds the dimension of the system, $n$.
      ne : int
          holds the number of entries in the  upper triangular part of $H.
      row : ndarray(ne)
          holds the row indices of the upper triangular part of $H$.
      col : ndarray(ne)
          holds the column indices of the  upper triangular part of $H$.
      options : dict, optional
          dictionary of control options (see ``sha.initialize``).

      **Returns:**

      m : int
          gives the minimum number of $(s^{(k)},y^{(k)})$ pairs that will
	  be needed to recover a good Hessian approximation.

   .. function:: sha.recover_matrix(ne, m, ls1, ls2, strans, ly1, ly2, ytrans, order=None)

      Compute the Hessian estimate $B$ from given difference pairs
      $(s^{(k)},y^{(k)})$.

      **Parameters:**

      ne : int
          holds the number of entries in the upper triangular part of
          the matrix $H$.
      m : int
          gives the number of $(s^{(k)},y^{(k)})$ pairs that are provided
	  to recover the Hessian approximation (see ``sha.analyse_matrix``
          for the minimum necessary).
      ls1 : int
          holds the leading (first) dimension of the array strans (below).
      ls2 : int
          holds the trailing (second) dimension of the array strans (below).
      strans : ndarray(ls1,ls2)
          holds the values of the vectors $\{s^{(k) T}\}$.
          Component [$k,i$] should hold $s_i^{(k)}$.
      ly1 : int
          holds the leading (first) dimension of the array ytrans (below).
      ly2 : int
          holds the trailing (second) dimension of the array ytrans (below).
      ytrans : ndarray(ly1,ly2)
          holds the values of the vectors $\{y^{(k) T}\}$.
          Component [$k,i$] should hold $y_i^{(k)}$.
      order : ndarray(m), optional
          holds the preferred order of access for the pairs
          $\{(s^{(k)},y^{(k)})\}$. The $k$-th component of order
          specifies the row number of strans and ytrans that will be
          used as the $k$-th most favoured. order need not be set
          if the natural order, $k, k = 1,...,$ m, is desired, and this
          case order should be None.

      **Returns:**

      val : ndarray(ne)
          holds the values of the nonzeros in the upper triangle of the matrix
          $B$ in the same order as specified in the sparsity pattern in
          ``sha.analyse_matrix``.

   .. function:: [optional] sha.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
            the return status.  Possible values are:

            * **0**

              The call was successful.

            * **1**

              Insufficient data pairs $(s_i,y_i)$ have been provided, as m
              is too small. The returned $B$ is likely not fully accurate.

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

              The restriction n > 0 or nz >= 0 or
              $\leq$ row[i] $\leq$ col[i] $\leq$ n has been violated.

            * **-31**

              The call to ``sha_estimate`` was not preceded by a call to
              ``sha_analyse_matrix``.
          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          max_degree : int
             the maximum degree in the adgacency graph.
          differences_needed : int
             the number of differences that will be needed.
          max_reduced_degree : int
             the maximum reduced degree in the adgacency graph.
          approximation_algorithm_used : int
             the approximation algorithm actually used.
          bad_row : int
             a failure occured when forming the bad_row-th row
             (0 = no failure).

   .. function:: sha.terminate()

     Deallocate all internal private storage.
