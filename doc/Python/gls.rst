GLS
===

.. module:: galahad.gls

.. include:: gls_intro.rst

functions
---------

   .. function:: gls.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          lp : int
             Unit for error messages.
          wp : int
             Unit for warning messages.
          mp : int
             Unit for monitor output.
          ldiag : int
             Controls level of diagnostic output.
          btf : int
             Minimum block size for block-triangular form (BTF). Set to
             $n$ to avoid using BTF.
          maxit : int
             Maximum number of iterations.
          factor_blocking : int
             Level 3 blocking in factorize.
          solve_blas : int
             Switch for using Level 1 or 2 BLAS in solve.
          la : int
             Initial size for real array for the factors.
          la_int : int
             Initial size for integer array for the factors.
          maxla : int
             Maximum size for real array for the factors.
          pivoting : int
             Controls pivoting: Number of columns searched. Zero for
             Markowitz.
          fill_in : int
             Initially fill_in * ne space allocated for factors.
          multiplier : float
             Factor by which arrays sizes are to be increased if they
             are too small.
          reduce : float
             if previously allocated internal workspace arrays are
             greater than reduce times the currently required sizes,
             they are reset to current requirment.
          u : float
             Pivot threshold.
          switch_full : float
             Density for switch to full code.
          drop : float
             Drop tolerance.
          tolerance : float
             anything < this is considered zero.
          cgce : float
             Ratio for required reduction using IR.
          diagonal_pivoting : bool
             Set to 0 for diagonal pivoting.
          struct_abort : bool
             abort if $A$ is structurally singular.

   .. function:: [optional] gls.information()

      Provide optional output information

      **Returns:**

      ainfo : dict
         dictionary containing output information from the analysis phase:
          flag : int
             Flags success or failure case.
          more : int
             More information on failure.
          len_analyse : int
             Size for analysis.
          len_factorize : int
             Size for factorize.
          ncmpa : int
             Number of compresses.
          rank : int
             Estimated rank.
          drop : int
             Number of entries dropped.
          struc_rank : int
             Structural rank of matrix.
          oor : int
             Number of indices out-of-range.
          dup : int
             Number of duplicates.
          stat : int
             STAT value after allocate failure.
          lblock : int
             Size largest non-triangular block.
          sblock : int
             Sum of orders of non-triangular blocks.
          tblock : int
             Total entries in all non-tringular blocks.
          ops : float
             Number of operations in elimination.
      finfo : dict
         dictionary containing output information from the factorization phase:
          flag : int
             Flags success or failure case.
          more : int
             More information on failure.
          size_factor : int
             Number of words to hold factors.
          len_factorize : int
             Size for subsequent factorization.
          drop : int
             Number of entries dropped.
          rank : int
             Estimated rank.
          stat : int
             Status value after allocate failure.
          ops : float
             Number of operations in elimination.
      sinfo : dict
         dictionary containing output information from the solve phase:
          flag : int
             Flags success or failure case.
          more : int
             More information on failure.
          stat : int
             Status value after allocate failure.

   .. function:: gls.finalize()

     Deallocate all internal private storage.
