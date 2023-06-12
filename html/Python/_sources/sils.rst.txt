SILS
====

.. module:: galahad.sils

.. include:: sils_intro.rst

functions
---------

   .. function:: sils.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          ICNTL : int
             ``MA27`` internal integer controls.
          lp : int
             Unit for error messages.
          wp : int
             Unit for warning messages.
          mp : int
             Unit for monitor output.
          sp : int
             Unit for statistical output.
          ldiag : int
             Controls level of diagnostic output.
          la : int
             Initial size for real array for the factors. If less than
             nrlnec, default size used.
          liw : int
             Initial size for integer array for the factors. If less
             than nirnec, default size used.
          maxla : int
             Max. size for real array for the factors.
          maxliw : int
             Max. size for integer array for the factors.
          pivoting : int
             Controls pivoting. Possible values are

             * **1**

               Numerical pivoting will be performed.

             * **2**

               No pivoting will be performed and an error exit will
               occur  immediately a pivot sign change is detected.

             * **3**

               No pivoting will be performed and an error exit will
               occur if a  zero pivot is detected.

             * **4**

               No pivoting is performed but pivots are changed to all be.

          nemin : int
             Minimum number of eliminations in a step (unused).
          factorblocking : int
             Level 3 blocking in factorize (unused).
          solveblocking : int
             Level 2 and 3 blocking in solve.
          thresh : int
             Controls threshold for detecting full rows in analyse,
             registered as percentage of N, 100 Only fully dense rows
             detected (default).
          ordering : int
             Controls ordering: Possible values are

             * **0**

               AMD using HSL's MC47

             * **1**

               User defined

             * **2**

               AMD using HSL's MC50

             * **3**

               Minimum degreee as in HSL's MA57

             * **4**

               Metis_nodend ordering

             * **5**

               Ordering chosen depending on matrix characteristics.
               At the moment choices are HSL's MC50 or Metis_nodend

             * **>5**

               Presently equivalent to 5 but may chnage.

          scaling : int
             Controls scaling: Possible values are

             * **0**

               No scaling

             * **>0**

               Scaling using HSL's MC64 but may change for > 1.

          CNTL : float
             MA27 internal real controls.
          multiplier : float
             Factor by which arrays sizes are to be increased if they
             are too small.
          reduce : float
             If previously allocated internal workspace arrays are
             greater than reduce times the currently required sizes,
             they are reset to current requirment.
          u : float
             Pivot threshold.
          static_tolerance : float
             used for setting static pivot level.
          static_level : float
             used for switch to static.
          tolerance : float
             Anything less than this is considered zero.
          convergence : float
             used to monitor convergence in iterative refinement.

   .. function:: [optional] sils.information()

      Provide optional output information

      **Returns:**

      ainfo : dict
         dictionary containing output information from the analysis phase:
          flag : int
             Flags success or failure case.
          more : int
             More information on failure.
          nsteps : int
             Number of elimination steps.
          nrltot : int
             Size for a without compression.
          nirtot : int
             Size for iw without compression.
          nrlnec : int
             Size for a with compression.
          nirnec : int
             Size for iw with compression.
          nrladu : int
             Number of reals to hold factors.
          niradu : int
             Number of integers to hold factors.
          ncmpa : int
             Number of compresses.
          oor : int
             Number of indices out-of-range.
          dup : int
             Number of duplicates.
          maxfrt : int
             Forecast maximum front size.
          stat : int
             STAT value after allocate failure.
          faulty : int
             legacy component, now not used.
          opsa : float
             Anticipated number of operations in assembly.
          opse : float
             Anticipated number of operations in elimination.
      finfo : dict
         dictionary containing output information from the factorization phase:
          flag : int
             Flags success or failure case.
          more : int
             More information on failure.
          maxfrt : int
             Largest front size.
          nebdu : int
             Number of entries in factors.
          nrlbdu : int
             Number of reals that hold factors.
          nirbdu : int
             Number of integers that hold factors.
          nrltot : int
             Size for a without compression.
          nirtot : int
             Size for iw without compression.
          nrlnec : int
             Size for a with compression.
          nirnec : int
             Size for iw with compression.
          ncmpbr : int
             Number of compresses of real data.
          ncmpbi : int
             Number of compresses of integer data.
          ntwo : int
             Number of 2x2 pivots.
          neig : int
             Number of negative eigenvalues.
          delay : int
             Number of delayed pivots (total).
          signc : int
             Number of pivot sign changes when options[`pivoting`]=3.
          nstatic : int
             Number of static pivots chosen.
          modstep : int
             First pivot modification when options[`pivoting`]=4.
          rank : int
             Rank of original factorization.
          stat : int
             STAT value after allocate failure.
          faulty : int
             legacy component, now not used.
          step : int
             legacy component, now not used.
          opsa : float
             # operations in assembly.
          opse : float
             number of operations in elimination.
          opsb : float
             Additional number of operations for BLAS.
          maxchange : float
             Largest options[`pivoting`]=4 modification.
          smin : float
             Minimum scaling factor.
          smax : float
             Maximum scaling factor.
      sinfo : dict
         dictionary containing output information from the solve phase:
          flag : int
             Flags success or failure case.
          stat : int
             STAT value after allocate failure.
          cond : float
             Condition number of matrix (category 1 eqs).
          cond2 : float
             Condition number of matrix (category 2 eqs).
          berr : float
             Backward error for the system (category 1 eqs).
          berr2 : float
             Backward error for the system (category 2 eqs).
          error : float
             Estimate of forward error.


   .. function:: sils.finalize()

     Deallocate all internal private storage.
