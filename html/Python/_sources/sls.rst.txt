SLS
===

.. module:: galahad.sls

The ``sls`` package 
**solves dense or sparse symmetric systems of linear equations**
using variants of Gaussian elimination.
Given a sparse symmetric matrix $A = \{ a_{ij} \}_{n \times n}$, and an
$n$-vector $b$ or a matrix $B = \{ b_{ij} \}_{n \times r}$, this
function solves the system $A x = b$ or the system $A X = B$ . 
The matrix $A$ need not be definite.

The method provides a common interface to a variety of well-known
solvers from HSL and elsewhere. Currently supported solvers include
``MA27/SILS``, ``HSL_MA57``, ``HSL_MA77`` , ``HSL_MA86``,
``HSL_MA87`` and ``HSL_MA97`` from {HSL},
``SSIDS`` from {SPRAL},
``MUMPS`` from Mumps Technologies,
``PARDISO`` both from the Pardiso Project and Intel's MKL,
``PaStiX`` from Inria, and
``WSMP`` from the IBM alpha Works, 
as well as ``POTR``, ``SYTR`` and ``SBTR`` from LAPACK.
Note that, with the exception of ``SSIDS`` and the Netlib
reference LAPACK codes,
**the solvers themselves do not form part of this package and
must be obtained/linked to separately.**
Dummy instances are provided for solvers that are unavailable.
Also note that additional flexibility may be obtained by calling the
solvers directly rather that via this package.

terminology
-----------

The solvers used each produce an $L D L^T$ factorization of
$A$ or a perturbation thereof, where $L$ is a permuted
lower triangular matrix and $D$ is a block diagonal matrix with
blocks of order 1 and 2. It is convenient to write this factorization in
the form
$$A + E = P L D L^T P^T,$$
where $P$ is a permutation matrix and $E$ is any diagonal
perturbation introduced.

supported solvers
-----------------

The key features of the external solvers supported by ``sls`` are
given in the following table:

.. list-table:: External solver characteristics
   :widths: 50 50 40 40 80
   :header-rows: 1

   * - solver 
     - factorization 
     - indefinite $A$ 
     - out-of-core 
     - parallelised
   * - ``SILS/MA27`` 
     - multifrontal 
     - yes 
     - no 
     - no
   * - ``HSL_MA57`` 
     - multifrontal 
     - yes 
     - no 
     - no
   * - ``HSL_MA77`` 
     - multifrontal 
     - yes 
     - yes 
     - OpenMP core
   * - ``HSL_MA86`` 
     - left-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``HSL_MA87`` 
     - left-looking 
     - no 
     - no 
     - OpenMP fully
   * - ``HSL_MA97`` 
     - multifrontal 
     - yes 
     - no 
     - OpenMP core
   * - ``SSIDS`` 
     - multifrontal 
     - yes 
     - no 
     - CUDA core
   * - ``MUMPS`` 
     - multifrontal 
     - yes 
     - optionally 
     - MPI
   * - ``PARDISO`` 
     - left-right-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``MKL_PARDISO`` 
     - left-right-looking 
     - yes 
     - optionally 
     - OpenMP fully
   * - ``PaStix`` 
     - left-right-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``WSMP`` 
     - left-right-looking 
     - yes 
     - no 
     - OpenMP fully
   * - ``POTR`` 
     - dense 
     - no 
     - no 
     - with parallel LAPACK
   * - ``SYTR`` 
     - dense 
     - yes 
     - no 
     - with parallel LAPACK
   * - ``PBTR`` 
     - dense band 
     - no 
     - no 
     - with parallel LAPACK

method
------

Variants of sparse Gaussian elimination are used.
See Section 4 of $GALAHAD/doc/sls.pdf for a brief description of the
method employed and other details.

The solver ``SILS`` is available as part of GALAHAD and relies on
the HSL Archive package ``MA27``. To obtain HSL Archive packages, see

- http://hsl.rl.ac.uk/archive/ .

The solvers
``HSL_MA57``,
``HSL_MA77``,
``HSL_MA86``,
``HSL_MA87``
and
``HSL_MA97``, the ordering packages
``MC61`` and ``HSL_MC68``, and the scaling packages
``HSL_MC64`` and ``MC77``
are all part of HSL 2011.
To obtain HSL 2011 packages, see

- http://hsl.rl.ac.uk .

The solver ``SSIDS`` is from the SPRAL sparse-matrix collection,
and is available as part of GALAHAD.

The solver ``MUMPS`` is available from Mumps Technologies in France, and 
version 5.5.1 or above is sufficient.
To obtain ``MUMPS``, see

- https://mumps-solver.org .

The solver ``PARDISO`` is available from the Pardiso Project;
version 4.0.0 or above is required.
To obtain ``PARDISO``, see

- http://www.pardiso-project.org/ .

The solver ``MKL PARDISO`` is available as part of Intel's oneAPI Math Kernel
Library (oneMKL).
To obtain this version of ``PARDISO``, see

- https://software.intel.com/content/www/us/en/develop/tools/oneapi.html .

The solver ``PaStix`` is available from Inria in France, and 
version 6.2 or above is sufficient.
To obtain ``PaStiX``, see

- https://solverstack.gitlabpages.inria.fr/pastix .

The solver ``WSMP`` is available from the IBM alpha Works;
version 10.9 or above is required.
To obtain ``WSMP``, see

- http://www.alphaworks.ibm.com/tech/wsmp .

The solvers ``POTR``, ``SYTR`` and ``PBTR``,
are available as
``S/DPOTRF/S``,
``S/DSYTRF/S`` and ``S/DPBTRF/S``
as part of LAPACK. Reference versions
are provided by GALAHAD, but for good performance
machined-tuned versions should be used.

Explicit sparsity re-orderings are obtained by calling the HSL package
``HSL_MC68``.
Both this, ``HSL_MA57`` and ``PARDISO`` rely optionally
on the ordering package ``MeTiS`` (version 4) from the Karypis Lab.
To obtain ``METIS``, see

- http://glaros.dtc.umn.edu/gkhome/views/metis/ .

Bandwidth, Profile and wavefront reduction is supported by
calling HSL's ``MC61``.

The methods used are described in the user-documentation for

HSL 2011, A collection of Fortran codes for large-scale scientific computation (2011). 

- http://www.hsl.rl.ac.uk

and papers

E. Agullo, P. R. Amestoy, A. Buttari, J.-Y. L'Excellent, A. Guermouche 
and F.-H. Rouet,
"Robust memory-aware mappings for parallel multifrontal factorizations".
SIAM Journal on Scientific Computing, \b 38(3) (2016), C256--C279,

P. R. Amestoy, I. S. Duff, J. Koster and J.-Y. L'Excellent.
"A fully asynchronous multifrontal solver using distributed 
dynamic scheduling".
SIAM Journal on Matrix Analysis and Applications \b 23(1) (2001) 15-41,

A. Gupta,
"WSMP: Watson Sparse Matrix Package Part I - direct
solution of symmetric sparse systems".
IBM Research Report RC 21886, IBM T. J. Watson Research Center,
NY 10598, USA (2010),

P. Henon, P. Ramet and J. Roman,
"PaStiX: A High-Performance Parallel Direct Solver for Sparse Symmetric 
Definite Systems".
Parallel Computing, \b 28(2) (2002) 301--321,

J.D. Hogg, E. Ovtchinnikov and J.A. Scott. 
"A sparse symmetric indefinite direct solver for GPU architectures".
ACM Transactions on Mathematical Software \b 42(1) (2014), Article 1,

O. Schenk and K. Gartner,
"Solving Unsymmetric Sparse Systems of Linear Equations with PARDISO".
Journal of Future Generation Computer Systems \b, 20(3) (2004) 475--487,
and

O. Schenk and K. Gartner,
"On fast factorization pivoting methods for symmetric indefinite systems".
Electronic Transactions on Numerical Analysis \b 23 (2006) 158--179.



matrix storage
--------------

The **symmetric** $n$ by $n$ matrix $A$ may 
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $A$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $A$ is
symmetric, only the lower triangular part (that is the part
$A_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array A_val
will hold the value $A_{ij}$ (and, by symmetry, $A_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string A_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays A_row and A_col and real array A_val,
respectively, while the number of nonzeros is recorded as
A_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string A_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, and values
$A_{ij}$ of the  entries in the i-th row are stored in components
l = A_ptr(i), ..., A_ptr(i+1)-1 of the
integer array A_col, and real array A_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $A$ is diagonal (i.e., $A_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$A_{ii}$, $0 \leq i \leq n-1$ need be stored, 
and the first n components of the array A_val may be used for the purpose.
The string A_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $A$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of A_val.
The string A_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $A$ is the identity matrix, no values need be stored.
The string A_type = 'identity' should be specified.

The *zero matrix* format:
The same is true if $A$ is the zero matrix, but now
the string A_type = 'zero' or 'none' should be specified.


functions
---------

   .. function:: sls.initialize(solver)

      Set default option values and initialize private data

      **Parameters:**

      solver : str
        the name of the solver required to solve $Ax=b$. 
        It should be one of 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 
        'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl pardiso', 
        'pastix', 'wsmp', 'potr', 'sytr' or 'pbtr';
        lower or upper case variants are allowed.

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          warning : int
             unit for warning messages.
          out : int
             general output occurs on stream out.
          statistics : int
             unit for statistical output.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output.

             * **1**

               gives a summary of the process.

             * **>=2**

               gives increasingly verbose (debugging) output.

          print_level_solver : int
             controls level of diagnostic output from external solver.
          bits : int
             number of bits used in architecture.
          block_size_kernel : int
             the target blocksize for kernel factorization.
          block_size_elimination : int
             the target blocksize for parallel elimination.
          blas_block_size_factorize : int
             level 3 blocking in factorize.
          blas_block_size_solve : int
             level 2 and 3 blocking in solve.
          node_amalgamation : int
             a child node is merged with its parent if they both
             involve fewer than node_amalgamation eliminations.
          initial_pool_size : int
             initial size of task-pool arrays for parallel elimination.
          min_real_factor_size : int
             initial size for real array for the factors and other data.
          min_integer_factor_size : int
             initial size for integer array for the factors and other
             data.
          max_real_factor_size : long
             maximum size for real array for the factors and other data.
          max_integer_factor_size : long
             maximum size for integer array for the factors and other
             data.
          max_in_core_store : long
             amount of in-core storage to be used for out-of-core
             factorization.
          array_increase_factor : float
             factor by which arrays sizes are to be increased if they
             are too small.
          array_decrease_factor : float
             if previously allocated internal workspace arrays are
             greater than array_decrease_factor times the currently
             required sizes, they are reset to current requirements.
          pivot_control : int
             pivot control. Possible values are

             * **1**

               Numerical pivoting will be performed.

             * **2**

               No pivoting will be performed and an error exit will
               occur immediately a pivot sign change is detected.

             * **3**

               No pivoting will be performed and an error exit will

                 occur if a zero pivot is detected.
             * **4**

               No pivoting is performed but pivots are changed to all
               be  positive.

          ordering : int
             controls ordering (ignored if explicit PERM argument
             present). Possible values are

             * **<0**

               chosen by the specified solver with its  own
               ordering-selected value -ordering

             * **0**

               chosen package default (or the AMD ordering if no
               package  default)

             * **1**

               Approximate minimum degree (AMD) with provisions for
               "dense"  rows/cols

             * **2**

               Minimum degree

             * **3**

               Nested disection

             * **4**

               indefinite ordering to generate a combination of 1x1
               and 2x2 pivots

             * **5**

               Profile/Wavefront reduction

             * **6**

               Bandwidth reduction

             * **>6**

               ordering chosen depending on matrix characteristics
               (not yet implemented).
          full_row_threshold : int
             controls threshold for detecting full rows in analyse,
             registered as percentage of matrix order. If 100, only
             fully dense rows detected (defa.
          row_search_indefinite : int
             number of rows searched for pivot when using indefinite
             ordering.
          scaling : int
             controls scaling (ignored if explicit SCALE argument present).
             Possible values are

             * **<0**

               chosen by the specified solver with its  own
               scaling-selected value -scaling

             * **0**

               No scaling

             * **1**

               Scaling using HSL's MC64

             * **2**

               Scaling using HSL's MC77 based on the row one-norm

             * **3**

               Scaling using HSL's MC77 based on the row
               infinity-norm.

          scale_maxit : int
             the number of scaling iterations performed (default 10
             used if ``scale_maxit`` < 0).
          scale_thresh : float
             the scaling iteration stops as soon as the row/column
             norms are less than 1+/-.scale_thresh.
          relative_pivot_tolerance : float
             pivot threshold.
          minimum_pivot_tolerance : float
             smallest permitted relative pivot threshold.
          absolute_pivot_tolerance : float
             any pivot small than this is considered zero.
          zero_tolerance : float
             any entry smaller than this is considered zero.
          zero_pivot_tolerance : float
             any pivot smaller than this is considered zero for
             positive-definite sol.
          negative_pivot_tolerance : float
             any pivot smaller than this is considered to be negative
             for p-d solvers.
          static_pivot_tolerance : float
             used for setting static pivot level.
          static_level_switch : float
             used for switch to static.
          consistency_tolerance : float
             used to determine whether a system is consistent when
             seeking a Fredholm alternative.
          max_iterative_refinements : int
             maximum number of iterative refinements allowed.
          acceptable_residual_relative : float
             refinement will cease as soon as the residual ||Ax-b||
             falls below max( acceptable_residual_relative * ||b||,
             acceptable_residual_absolute.
          acceptable_residual_absolute : float
             see acceptable_residual_relative.
          multiple_rhs : bool
             set ``multiple_rhs`` to ``true.`` if there is possibility
             that the solver will be required to solve systems with
             more than one right-hand side. More efficient execution
             may be possible when ``multiple_rhs`` = ``false.``.
          generate_matrix_file : bool
             if ``generate_matrix_file`` is ``true.`` if a file
             describing the current matrix is to be generated.
          matrix_file_device : int
             specifies the unit number to write the input matrix (in
             co-ordinate form.
          matrix_file_name : str
             name of generated matrix file containing input problem.
          out_of_core_directory : str
             directory name for out of core factorization and
             additional real workspace in the indefinite case,
             respectively.
          out_of_core_integer_factor_file : str
             out of core superfile names for integer and real factor
             data, real works and additional real workspace in the
             indefinite case, respectively.
          out_of_core_real_factor_file : str
             see out_of_core_integer_factor_file.
          out_of_core_real_work_file : str
             see out_of_core_integer_factor_file.
          out_of_core_indefinite_file : str
             see out_of_core_integer_factor_file.
          out_of_core_restart_file : str
             see out_of_core_integer_factor_file.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.

   .. function:: sls.analyse_matrix(n, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage and compute sparsity-based 
      reorderings prior to factorization.

      **Parameters:**

      n : int
          holds the dimension of the system, $n$.
      A_type : string
          specifies the symmetric storage scheme used for the matrix $A$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in the  lower triangular part of
          $A$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      A_row : ndarray(A_ne)
          holds the row indices of the lower triangular part of $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of the  lower triangular part of
          $A$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      A_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $A$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``sls.initialize``).

   .. function:: sls.factorize_matrix(a_ne, A_val)

      Factorize the matrix $A$ prior to solution.

      **Parameters:**

      a_ne : int
          holds the number of entries in the lower triangular part of 
          the matrix $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the lower triangle of the matrix
          $A$ in the same order as specified in the sparsity pattern in 
          ``sls.load``.

   .. function:: sls.solve_system(n, b)

      Given the factors of $A$, solve the system of linear equations $Ax=b$.

      **Parameters:**

      n : int
          holds the dimension of the system, $n$.
          holds the number of variables.
      b : ndarray(n)
          holds the values of the right-hand side vector $b$

      **Returns:**

      x : ndarray(n)
          holds the values of the solution $x$ after a successful call.

   .. function:: sls.partial_solve_system(part, n, b)

      Given the factorization $A = PLDUP^T$ where $U = L^T$, 
      solve the system of linear equations 
      $Mx=b$, where $M$ is one of $PL,$ $D,$ $UP^T$ or $S = PL\sqrt{D}$.

      **Parameters:**

      part : str
          single character code that specifies which factor $M$ applies. 
          Possible values are:

          * **L**

            solve $PL x = b $

          * **D**

            solve $D x = b $

          * **U**

            solve $UP^T x = b $

          * **S**

            solve $S x = b $

          lower or upper case variants are allowed.
      n : int
          holds the dimension of the system, $n$.
          holds the number of variables.
      b : ndarray(n)
          holds the values of the right-hand side vector $b$

      **Returns:**

      x : ndarray(n)
          holds the values of the solution $x$ after a successful call.

   .. function:: [optional] sls.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
             reported return status. Possible values are

             * **0**

               The run was succesful.

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

               The restriction n > 0 or requirement that type contains
               its relevant string 'dense', 'coordinate', 'sparse_by_rows',
               'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
               has been violated.

             * **-20**

               The matrix $A$ is not positive definite while the factorization
               solver used expected it to be.

             * **-29**

               A requested option is unavailable.

             * **-31**

               The input order is not a permutation or is faulty
               in some other way.

             * **-32**

               more than options['max_integer_factor_size']
               integer space required for the factors.

             * **-33**

               more than options['max_real_factor_size']
               real space required for the factors.

             * **-40**

               It is not possible to alter the diagonals.

             * **-41**

               there is no access to the permutation or pivot sequence used.

             * **-42**

               there is no access to diagonal perturbations.

             * **-43**

               A direct-access file error has occurred.

             * **-50**

               A solver-specific error; see the solver's info parameter.

             * **-101**

               An unknown solver has been requested.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          more_info : int
             further information on failure.
          entries : int
             number of entries.
          out_of_range : int
             number of indices out-of-range.
          duplicates : int
             number of duplicates.
          upper : int
             number of entries from the strict upper triangle.
          missing_diagonals : int
             number of missing diagonal entries for an
             allegedly-definite matrix.
          max_depth_assembly_tree : int
             maximum depth of the assembly tree.
          nodes_assembly_tree : int
             nodes in the assembly tree (= number of elimination steps).
          real_size_desirable : long
             desirable or actual size for real array for the factors
             and other data.
          integer_size_desirable : long
             desirable or actual size for integer array for the factors
             and other dat.
          real_size_necessary : long
             necessary size for real array for the factors and other
             data.
          integer_size_necessary : long
             necessary size for integer array for the factors and other
             data.
          real_size_factors : long
             predicted or actual number of reals to hold factors.
          integer_size_factors : long
             predicted or actual number of integers to hold factors.
          entries_in_factors : long
             number of entries in factors.
          max_task_pool_size : int
             maximum number of tasks in the factorization task pool.
          max_front_size : int
             forecast or actual size of largest front.
          compresses_real : int
             number of compresses of real data.
          compresses_integer : int
             number of compresses of integer data.
          two_by_two_pivots : int
             number of 2x2 pivots.
          semi_bandwidth : int
             semi-bandwidth of matrix following bandwidth reduction.
          delayed_pivots : int
             number of delayed pivots (total).
          pivot_sign_changes : int
             number of pivot sign changes if no pivoting is used
             successfully.
          static_pivots : int
             number of static pivots chosen.
          first_modified_pivot : int
             first pivot modification when static pivoting.
          rank : int
             estimated rank of the matrix.
          negative_eigenvalues : int
             number of negative eigenvalues.
          num_zero : int
             number of pivots that are considered zero (and ignored).
          iterative_refinements : int
             number of iterative refinements performed.
          flops_assembly : long
             anticipated or actual number of floating-point operations
             in assembly.
          flops_elimination : long
             anticipated or actual number of floating-point operations
             in elimination.
          flops_blas : long
             additional number of floating-point operations for BLAS.
          largest_modified_pivot : float
             largest diagonal modification when static pivoting or
             ensuring definiten.
          minimum_scaling_factor : float
             minimum scaling factor.
          maximum_scaling_factor : float
             maximum scaling factor.
          condition_number_1 : float
             esimate of the condition number of the matrix (category 1
             equations).
          condition_number_2 : float
             estimate of the condition number of the matrix (category 2
             equations).
          backward_error_1 : float
             esimate of the backward error (category 1 equations).
          backward_error_2 : float
             esimate of the backward error (category 2 equations).
          forward_error : float
             estimate of forward error.
          alternative : bool
             has an "alternative" y: A y = 0 and yT b > 0 been found
             when trying to solve A x = b ?.
          time : dict
             timings (see above).
          sils_ainfo : dict
             the output structure from sils.
          sils_finfo : dict
             see sils_ainfo.
          sils_sinfo : dict
             see sils_ainfo.
          ma57_ainfo : dict
             the output structure from ma57.
          ma57_finfo : dict
             see ma57_ainfo.
          ma57_sinfo : dict
             see ma57_ainfo.
          ma77_info : dict
             the output structure from ma77.
          ma86_info : dict
             the output structure from ma86.
          ma87_info : dict
             the output structure from ma87.
          ma97_info : dict
             the output structure from ma97.
          ssids_inform : dict
             the output structure from ssids.
          mc61_info : int
             the integer and real output arrays from mc61.
          mc61_rinfo : float
             see mc61_info.
          mc64_info : dict
             the output structure from mc64.
          mc68_info : dict
             the output structure from mc68.
          mc77_info : int
             the integer output array from mc77.
          mc77_rinfo : float
             the real output status from mc77.
          mumps_error : int
             the output scalars and arrays from mumps.
          mumps_info : int
             see pardiso_error.
          mumps_rinfo : float
             see pardiso_error.
          pardiso_error : int
             the output scalars and arrays from pardiso.
          pardiso_IPARM : int
             see pardiso_error.
          pardiso_DPARM : float
             see pardiso_error.
          mkl_pardiso_error : int
             the output scalars and arrays from mkl_pardiso.
          mkl_pardiso_IPARM : int
             see mkl_pardiso_error.
          pastix_info : int
             the output scalar from pastix.
          wsmp_error : int
             the output scalars and arrays from wsmp.
          wsmp_iparm : int
             see wsmp_error.
          wsmp_dparm : float
             see wsmp_error.
          lapack_error : int
             the output scalars and arrays from LAPACK routines.
          time : dict
             dictionary containing timing information:
               total : float
                  the total cpu time spent in the package.
               analyse : float
                  the total cpu time spent in the analysis phase.
               factorize : float
                  the total cpu time spent in the factorization phase.
               solve : float
                  the total cpu time spent in the solve phases.
               order_external : float
                  the total cpu time spent by the external solver in the
                  ordering phase.
               analyse_external : float
                  the total cpu time spent by the external solver in the
                  analysis phase.
               factorize_external : float
                  the total cpu time spent by the external solver in the
                  factorization pha.
               solve_external : float
                  the total cpu time spent by the external solver in the
                  solve phases.
               clock_total : float
                  the total clock time spent in the package.
               clock_analyse : float
                  the total clock time spent in the analysis phase.
               clock_factorize : float
                  the total clock time spent in the factorization phase.
               clock_solve : float
                  the total clock time spent in the solve phases.
               clock_order_external : float
                  the total clock time spent by the external solver in the
                  ordering phase.
               clock_analyse_external : float
                  the total clock time spent by the external solver in the
                  analysis phase.
               clock_factorize_external : float
                  the total clock time spent by the external solver in the
                  factorization p.
               clock_solve_external : float
                  the total clock time spent by the external solver in the
                  solve phases.


   .. function:: sls.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/sls/Python/test_sls.py
   :code: python

This example code is available in $GALAHAD/src/sls/Python/test_sls.py .
