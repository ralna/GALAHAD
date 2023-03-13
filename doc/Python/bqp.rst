BQP
===

.. module:: galahad.bqp

The bqp package uses a **preconditioned, projected-gradient method** to solve a
given **bound-constrained convex quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
subject to the simple bounds
$$x_l \leq x \leq x_u,$$
where $H$ is a given $n$ by $n$ symmetric postive-semi-definite matrix,  
$g$ is a vector, $f$ is a scalar, and any of the components 
of the vectors $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/bqp.pdf for a brief description of the
method employed and other details.

matrix storage
--------------

The **symmetric** $n$ by $n$ matrix $H$ may also
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $H$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $H$ is
symmetric, only the lower triangular part (that is the part
$H_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array H_val
will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string H_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $H$,
its row index i, column index j and value $H_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays H_row and H_col and real array H_val,
respectively, while the number of nonzeros is recorded as
H_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string H_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries plus one.
The column indices j, $0 \leq j \leq i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices, 
this scheme almost always requires less storage than its predecessor.
The string H_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $H$ is diagonal (i.e., $H_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$H_{ii}$, $0 \leq i \leq n-1$ need be stored, 
and the first n components of the array H_val may be used for the purpose.
The string H_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $H$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of H_val.
The string H_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $H$ is the identity matrix, no values need be stored.
The string H_type = 'identity' should be specified.

The *zero matrix* format:
The same is true if $H$ is the zero matrix, but now
the string H_type = 'zero' or 'none' should be specified.


functions
---------

   .. function:: bqp.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          print_level : int
             the level of output required is specified by print_level.
             Possible values are

             * **<=0**

               gives no output,

             * **1**

               gives a one-line summary for every iteration.

             * **2**

               gives a summary of the inner iteration for each iteration.

             * **>=3**

               gives increasingly verbose (debugging) output.

          start_print : int
             on which iteration to start printing.
          stop_print : int
             on which iteration to stop printing.
          print_gap : int
             how many iterations between printing.
          maxit : int
             how many iterations to perform (-ve reverts to HUGE(1)-1).
          cold_start : int
             cold_start should be set to 0 if a warm start is required
             (with variable assigned according to B_stat, see below),
             and to any other value if the values given in prob.X
             suffice.
          ratio_cg_vs_sd : int
             the ratio of how many iterations use CG rather steepest
             descent.
          change_max : int
             the maximum number of per-iteration changes in the working
             set permitted when allowing CG rather than steepest
             descent.
          cg_maxit : int
             how many CG iterations to perform per BQP iteration (-ve
             reverts to n+1).
          sif_file_device : int
             the unit number to write generated SIF file describing the
             current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_p : float
             the required accuracy for the primal infeasibility.
          stop_d : float
             the required accuracy for the dual infeasibility.
          stop_c : float
             the required accuracy for the complementary slackness.
          identical_bounds_tol : float
             any pair of constraint bounds (x_l,x_u) that are closer
             than i dentical_bounds_tol will be reset to the average of
             their values.
          stop_cg_relative : float
             the CG iteration will be stopped as soon as the current
             norm of the preconditioned gradient is smaller than 
             max( stop_cg_relative * initial preconditioned gradient,
             stop_cg_absolute).
          stop_cg_absolute : float
             see stop_cg_relative.
          zero_curvature : float
             threshold below which curvature is regarded as zero.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve = no limit).
          exact_arcsearch : bool
             exact_arcsearch is True if an exact arcsearch is required,
             and False if approximation suffices.
          space_critical : bool
             if space_critical is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation times.
          deallocate_error_fatal : bool
             if deallocate_error_fatal is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          generate_sif_file : bool
             if generate_sif_file is True, a SIF file describing the
             current problem will be generated.
          sif_file_name : str
             name (max 30 characters) of generated SIF file containing
             input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sbls_control : dict
             control parameters for SBLS (see ``sbls.initialize``).

   .. function:: bqp.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      H_type : string
          specifies the symmetric storage scheme used for the Hessian $H$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      H_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      H_row : ndarray(H_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      H_col : ndarray(H_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries plus one,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``bqp.initialize``).

   .. function:: bqp.solve_qp(n, f, g, h_ne, H_val, x_l, x_u)

      Find a solution to the bound-constrained convex quadratic program 
      involving the quadratic objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      h_ne : int
          holds the number of entries in the lower triangular part of 
          the Hessian $H$.
      H_val : ndarray(h_ne)
          holds the values of the nonzeros in the lower triangle of the Hessian
          $H$ in the same order as specified in the sparsity pattern in 
          ``bqp.load``.
      x_l : ndarray(n)
          holds the values of the lower bounds $x_l$ on the variables.
          The lower bound on any component of $x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      x_u : ndarray(n)
          holds the values of the upper bounds $x_l$ on the variables.
          The upper bound on any component of $x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] bqp.information()

      Provide optional output information

      **Returns:**

      inform : dict
         dictionary containing output information:

          status : int
            return status.  Possible values are:

            * **0**

              The run was succesful.

            * **-1**

              An allocation error occurred. A message indicating the
              offending array is written on unit control['error'], and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-2**

              A deallocation error occurred.  A message indicating the
              offending array is written on unit control['error'] and
              the returned allocation status and a string containing
              the name of the offending array are held in
              inform['alloc_status'] and inform['bad_alloc'] respectively.

            * **-3**

              The restriction n > 0 or m > 0 or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
              has been violated.

            * **-4**

              The bound constraints are inconsistent.

            * **-7**

              The objective function appears to be unbounded from below
              on the feasible set.

            * **-9**

              The analysis phase of the factorization failed; the return
              status from the factorization package is given by
              inform['factor_status'].

            * **-10**

              The factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package failed; the return status
              from the factorization package is given by
              inform['factor_status'].

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-18**

              Too many iterations have been performed. This may happen if
              control['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

            * **-19**

              The CPU time limit has been reached. This may happen if
              control['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-20**

              The Hessian $H$ appears to be indefinite.

            * **-23** 

              An entry from the strict upper triangle of $H$ has been 
              specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          factorization_status : int
             status return from factorization.
          iter : int
             number of iterations required.
          cg_iter : int
             number of CG iterations required.
          obj : float
             current value of the objective function.
          norm_pg : float
             current value of the projected gradient.
          time : dict
             dictionary containing timing information:
               total : float
                  total time.
               analyse : float
                  time for the analysis phase.
               factorize : float
                  time for the factorization phase.
               solve : float
                  time for the linear solution phase.
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).

   .. function:: bqp.terminate()

     Deallocate all internal private storage.
