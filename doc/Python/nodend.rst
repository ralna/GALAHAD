NODEND
======

.. module:: galahad.nodend

.. include:: nodend_intro.rst

.. include:: nodend_storage.rst

functions
---------

   .. function:: nodend.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          version : str
            specify the version of METIS to be used. Possible values
            are 4.0, 5.1 and 5.2.
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required is specified by print_level.
          no_metis_4_use_5_instead : bool
             if `` no_metis_4_use_5_instead`` is True, and METIS 4.0 is 
             not availble, use Metis 5.2 instead.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          metis4_ptype : int
            the partitioning method employed. 0 = multilevel recursive 
             bisectioning: 1 = multilevel k-way partitioning
          metis4_ctype : int
            the matching scheme to be used during coarsening: 
            1 = random matching, 
            2 = heavy-edge matching, 3 = sorted heavy-edge matching, and
            4 = k-way sorted heavy-edge matching.
          metis4_itype : int
            the algorithm used during initial partitioning: 
            1 = edge-based region growing and 2 = node-based region growing.
          metis4_rtype : int
            the algorithm used for refinement: 
            1 = two-sided node Fiduccia-Mattheyses (FM) refinement, and
            2 = one-sided node FM refinement.
          metis4_dbglvl : int
            the amount of progress/debugging information printed: 
            0 = nothing, 1 = timings, and $>$ 1 increasingly more.
          metis4_oflags : int
            select whether or not to compress the graph, and to order 
            connected components separately: 0 = do neither, 
            1 = try to compress the graph, 
            2 = order each connected component separately, and 3 = do both.
          metis4_pfactor : int
            the minimum degree of the vertices that will be ordered last. 
            More specifically, any vertices with a degree greater than 
            0.1 metis4_pfactor times the average degree are removed from
            the graph, an ordering of the rest of the vertices is computed, 
            and an overall ordering is computed by ordering the removed 
            vertices at the end of the overall ordering. Any value
            smaller than 1 means that no vertices will be ordered last.
          metis4_nseps : int
            the number of different separators that the algorithm will 
            compute at each level of nested dissection.
          metis5_ptype : int
            the partitioning method. The value 0 gives multilevel recursive 
            bisectioning, while 1 corresponds to multilevel $k$-way 
            partitioning.
          metis5_objtype : int
            the type of the objective. Currently the only and default value
            metis5_objtype = 2, specifies node-based nested dissection, 
            and any invalid value will be replaced by this default.
          metis5_ctype : int
            the matching scheme to be used during coarsening: 
             0 = random matching, 
             and 1 = sorted heavy-edge matching.
          metis5_iptype : int
            the algorithm used during initial partitioning:
            2 = derive separators from edge cuts, and
            3 = grow bisections using a greedy node-based strategy.
          metis5_rtype : int
            the algorithm used for refinement: 2 = Two-sided node FM refinement,
             and 3 = One-sided node FM refinement.
          metis5_dbglvl : int
            the amount of progress/debugging information printed: 0 = nothing, 
            1 = diagnostics, 2 = plus timings, and $>$ 2 plus more.
          metis5_niparts : int
            the number of initial partitions used by MeTiS 5.2.
          metis5_niter : int
            the number of iterations used by the refinement algorithm.
          metis5_ncuts : int
            the number of different partitionings that it will compute: 
            -1 = not used.
          metis5_seed : int
            the seed for the random number generator.
          metis5_ondisk : int
            whether on-disk storage is used (0 = no, 1 = yes) by MeTiS 5.2.
          metis5_minconn : int
            specify that the partitioning routines should try to minimize 
            the maximum degree of the subdomain graph: 0 = no, 1 = yes, and 
            -1 = not used. 
          metis5_contig : int
            specify that the partitioning routines should try to produce 
            partitions that are contiguous: 0 = no, 1 = yes, and -1 = not used.
          metis5_compress : int
            specify that the graph should be compressed by combining together
            vertices that have identical adjacency lists: 0 = no, and 1 = yes.
          metis5_ccorder : int
            specify if the connected components of the graph should first be
            identified and ordered separately: 0 = no, and 1 = yes.
          metis5_pfactor : int
            the minimum degree of the vertices that will be ordered last.
            More specifically, any vertices with a degree greater than 
            0.1 metis4_pfactor times the average degree are removed from
            the graph, an ordering of the rest of the vertices is computed, 
            and an overall ordering is computed by ordering the removed 
            vertices at the end of the overall ordering.
          metis5_nseps : int
            the number of different separators that the algorithm will compute
            at each level of nested dissection.
          metis5_ufactor : int
            the maximum allowed load imbalance (1 + metis5_ufactor)/1000 
            among the partitions.
          metis5_dropedges : int
            will edges be dropped (0 = no, 1 = yes) by MeTiS 5.2.
          metis5_no2hop : int
            specify that the coarsening will not perform any 2–hop matchings 
            when the standard matching approach fails to sufficiently coarsen 
            the graph: 0 = no, and 1 = yes.
          metis5_twohop : int
            reserved for future use but ignored at present.
          metis5_fast : int
            reserved for future use but ignored at present.


   .. function:: nodend.order(n, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Find a row/colum permutation for sparse Cholesky-like factorization.

      **Parameters:**

      n : int
          holds the number of variables.
      A_type : string
          specifies the symmetric storage scheme used for $A$.
          It should be one of 'coordinate' or 'sparse_by_rows';
          lower or upper case variants are allowed. If A_type is not one
          of the supported values, the identity permutation will be returned.
      A_ne : int
          holds the number of entries in the  lower triangular part of
          $A$ in the sparse co-ordinate storage scheme. It need
          not be set for the sparse-by-row scheme.
      A_row : ndarray(A_ne)
          holds the row indices of the lower triangular part of $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          the sparse-by-row scheme, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of the  lower triangular part of
          $A$ in either of the supported storage schemes.
      A_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $A$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          coordinate scheme is used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``nodend.initialize``).

      **Returns:**

      perm : ndarray(n)
          holds the permutation array, so that the perm[$i$]-th rows and 
          columns in the permuted matrix $P A P^T$ correspond to those 
          labelled i in $A$, 0 $\leq$ i $\leq$ n-1.

   .. function:: [optional] nodend.information()

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

               One of the restrictions
               n $> 0$, A.n $> 0$ or A.ne $< 0$, for co-ordinate entry,
               or requirements that A.type contain its relevant string
               'coordinate' or 'sparse_by_rows', and
               options['version'] in one of '4.0', '5.1' or '5.2'
               has been violated.

             * **-26**

               The requested version of METIS is not available.

             * **-57**

               METIS has insufficient memory to continue.

             * **-71**

               An internal METIS error occurred.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          version : str
            specifies the version of METIS that was actually used.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent in the METIS package.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent in the METIS package.

example code
------------

.. include:: ../../src/nodend/Python/test_nodend.py
   :code: python

This example code is available in $GALAHAD/src/nodend/Python/test_nodend.py .
