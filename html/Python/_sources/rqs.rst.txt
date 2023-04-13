RQS
===

.. module:: galahad.rqs

The ``rqs`` package uses **matrix factorization** to find the 
global minimizer of a **regularized quadratic objective function**.
The aim is to minimize the regularized quadratic objective function
$$r(x) = f + g^T x + \frac{1}{2} x^T H x + \frac{\sigma}{p} \|x\|_{M}^p,$$ 
where the **weight** $\sigma > 0$, the **power** $p \geq 2$, the  vector $x$
may optionally  be required to satisfy **affine constraints** $A x = 0,$
and where the $M$-norm of $x$ is defined to be $\|x\|_{M} = \sqrt{x^T M x}$.

The matrix $M$ need not be provided in the commonly-occurring
$\ell_2$-regularization case for which $M = I$, the $n$ by $n$
identity matrix.

Factorization of matrices of the form $H + \lambda M$, or
$$\left(\begin{array}{cc} H + \lambda M & A^T \\ A & 0 \end{array}\right)$$
in cases where $A x = 0$ is imposed, for a succession
of scalars $\lambda$ will be required, so this package is most suited
for the case where such a factorization may be found efficiently. If
this is not the case, the package ``glrt`` may be preferred.

See Section 4 of $GALAHAD/doc/rqs.pdf for a brief description of the
method employed and other details.

matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$, if it is needed, may be presented
and stored in a variety of convenient input formats. 

*Dense* storage format:
The matrix $A$ is stored as a compact dense matrix by rows, that is,
the values of the entries of each row in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $n \ast i + j$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.
The string A_type = 'dense' should be specified.

*Dense by columns* storage format:
The matrix $A$ is stored as a compact dense matrix by columns, that is,
the values of the entries of each column in turn are
stored in order within an appropriate real one-dimensional array.
In this case, component $m \ast j + i$  of the storage array A_val
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1$.
The string A_type = 'dense_by_columns' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq i \leq m-1$,  $0 \leq j \leq n-1$,  are stored as the $l$-th 
components of the integer arrays A_row and A_col and real array A_val, 
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
The string A_type = 'coordinate' should be specified.

*Sparse row-wise storage* format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries.
The column indices j, $0 \leq j \leq n-1$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array A_col, and real array A_val, respectively.
For sparse matrices, this scheme almost always requires less storage than
its predecessor.
The string A_type = 'sparse_by_rows' should be specified.

*Sparse column-wise* storage format:
Once again only the nonzero entries are stored, but this time
they are ordered so that those in column j appear directly before those
in column j+1. For the j-th column of $A$ the j-th component of the
integer array A_ptr holds the position of the first entry in this column,
while A_ptr(n) holds the total number of entries.
The row indices i, $0 \leq i \leq m-1$, and values $A_{ij}$
of the  nonzero entries in the j-th columnsare stored in components
l = A_ptr(j), $\ldots$, A_ptr(j+1)-1, $0 \leq j \leq n-1$,
of the integer array A_row, and real array A_val, respectively.
As before, for sparse matrices, this scheme almost always requires less
storage than the co-ordinate format.
The string A_type = 'sparse_by_columns' should be specified.

The **symmetric** $n$ by $n$ matrices $H$ and, optionally. $M$ may also
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
while H_ptr(n) holds the total number of entries.
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

   .. function:: rqs.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
        dictionary containing default control options:
          error : int
             error and warning diagnostics occur on stream error.
          out : int
             general output occurs on stream out.
          problem : int
             unit to write problem data into file problem_file.
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

          dense_factorization : int
             should the problem be solved by dense factorization?
             Possible values are

             * **0**

               sparse factorization will be used

             * **1**

               dense factorization will be used

             * **other**

               the choice is made automatically depending on the

             dimension  and sparsity.

          new_h : int
             how much of $H$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1**

               values but not indices have changed

             * **2**

               values and indices have changed.

          new_m : int
             how much of $M$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1**

               values but not indices have changed

             * **2**

               values and indices have changed.

          new_a : int
             how much of $A$ has changed since the previous call.
             Possible values are 0 unchanged 1 values but not indices
             have changed 2 values and indices have changed.
          max_factorizations : int
             the maximum number of factorizations (=iterations)
             allowed. -ve implies no limit.
          inverse_itmax : int
             the number of inverse iterations performed in the "maybe
             hard" case.
          taylor_max_degree : int
             maximum degree of Taylor approximant allowed.
          initial_multiplier : float
             initial estimate of the Lagrange multipler.
          lower : float
             lower and upper bounds on the multiplier, if known.
          upper : float
             see lower.
          stop_normal : float
             stop when $| \|x\| - (multiplier/\sigma)^(1/(p-2)) | \leq$
             stop_normal *
             max$( \|x\|, (multiplier/\sigma)^(1/(p-2)) )$
          stop_hard : float
             stop when bracket on optimal multiplier <= stop_hard *
             max( bracket ends ).
          start_invit_tol : float
             start inverse iteration when bracket on optimal multiplier
             <= stop_start_invit_tol * max( bracket ends ).
          start_invitmax_tol : float
             start full inverse iteration when bracket on multiplier <=
             stop_start_invitmax_tol * max( bracket ends).
          use_initial_multiplier : bool
             ignore initial_multiplier?.
          initialize_approx_eigenvector : bool
             should a suitable initial eigenvector should be chosen or
             should a previous eigenvector may be used?.
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
          problem_file : str
             name of file into which to write problem data.
          symmetric_linear_solver : str
             symmetric (indefinite) linear equation solver.
          definite_linear_solver : str
             definite linear equation solver.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          ir_options : dict
             default control options for IR (see ``ir.initialize``).

   .. function:: rqs.load(n, H_type, H_ne, H_row, H_col, H_ptr, options=None)

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
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``rqs.initialize``).

   .. function:: rqs.load_m(n, M_type, M_ne, M_row, M_col, M_ptr, options=None)

      Import problem data for the scaling matrix $M$, if needed, 
      into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      M_type : string
          specifies the symmetric storage scheme used for the Hessian $H$.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none'; 
          lower or upper case variants are allowed.
      M_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      M_row : ndarray(M_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      M_col : ndarray(M_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      M_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``rqs.initialize``).

   .. function:: rqs.load_a(m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data for the constraint matrix $A$, if needed, 
      into internal storage prior to solution.

      **Parameters:**

      m : int
          holds the number of constraints.
      A_type : string
          specifies the unsymmetric storage scheme used for the Hessian $A$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in 
          $A$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      A_row : ndarray(A_ne)
          holds the row indices of $A$ in the sparse co-ordinate storage 
          scheme. It need not be set for any of the other schemes, 
          and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of $A$ in either the sparse co-ordinate, 
          or the sparse row-wise storage scheme. It need not be set when 
          the other storage schemes are used, and in this case can be None.
      A_ptr : ndarray(m+1)
          holds the starting position of $A$, as well as the total number 
          of entries, in the sparse row-wise storage scheme. 
          It need not be set when the other schemes are used, 
          and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``rqs.initialize``).

   .. function:: rqs.solve_problem(n, weight, power, f, g, h_ne, H_val, m_ne, M_val, m, A_ne, A_val)

      Find the global minimizer of the regularized quadratic objective 
      function $r(x)$ subject to the affine constraints.

      **Parameters:**

      n : int
          holds the number of variables.
      weight : float
          holds the strictly positive regularization weight, $\sigma$.
      power : float
          holds the regularization power, $p \geq 2$.
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
          ``rqs.load``.
      m_ne : int
          holds the number of entries in the lower triangular part of 
          the scaling matrix $M$ if it is not the identity matrix. 
          Otherwise it should be None.
      M_val : ndarray(m_ne)
          holds the values of the nonzeros in the lower triangle of the scaling
          matrix $M$ in the same order as specified in the sparsity pattern in 
          ``rqs.load_m`` if needed. Otherwise it should be None.
      m : int
          holds the number of constraints.
      a_ne : int
          holds the number of entries in the lower triangular part of 
          the constraint matrix $A$ if $m > 0$.
          Otherwise it should be None.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the lower triangle of the 
          constraint matrix $A$ in the same order as specified in the 
          sparsity pattern in ``rqs.load_a`` if needed. 
          Otherwise it should be None.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          affine constraints, if any.
          Absent if ``trs.load_a`` has not been called.

   .. function:: [optional] rqs.information()

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

              The restriction n > 0, m > 0, weight > 0 or 
              power $\geq$ 2, or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' 
              has been violated.

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

            * **-15** 

              $M$ does not appear to be strictly diagonally dominant

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-19**

              The CPU time limit has been reached. This may happen if
              control['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-23** 

              An entry from the strict upper triangle of $H$ has been 
              specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          factorizations : int
             the number of factorizations performed.
          max_entries_factors : long
             the maximum number of entries in the factors.
          len_history : int
             the number of $(\|x\|_M,\lambda)$ pairs in the history.
          obj : float
             the value of the quadratic function.
          obj_regularized : float
             the value of the regularized quadratic function.
          x_norm : float
             the $M$-norm of $x$, $\|x\|_M$.
          multiplier : float
             the Lagrange multiplier corresponding to the
             regularization.
          pole : float
             a lower bound $\max(0,-\lambda_1)$, where $\lambda_1$
             is the left-most eigenvalue of $(H,M)$.
          dense_factorization : bool
             was a dense factorization used?.
          hard_case : bool
             has the hard case occurred?.
          time : dict
             dictionary containing timing information:
               total : float
                  total CPU time spent in the package.
               assemble : float
                  CPU time spent building $H + \lambda M$.
               analyse : float
                  CPU time spent reordering $H + \lambda M$ prior to
                  factorization.
               factorize : float
                  CPU time spent factorizing $H + \lambda M$.
               solve : float
                  CPU time spent solving linear systems inolving
                  $H + \lambda M$.
               clock_total : float
                  total clock time spent in the package.
               clock_assemble : float
                  clock time spent building $H + \lambda M$.
               clock_analyse : float
                  clock time spent reordering $H + \lambda M$ prior to
                  factorization.
               clock_factorize : float
                  clock time spent factorizing $H + \lambda M$.
               clock_solve : float
                  clock time spent solving linear systems inolving
                  $H + \lambda M$.
          history : dict
             dictionary containing information recording the history of the iterates:
               lambda : float
                  the value of $\lambda$.
               x_norm : float
                  the corresponding value of $\|x(\lambda)\|_M$.

          sls_inform : dict
             inform parameters for SLS (see ``sbls.information``).
          ir_inform : dict
             inform parameters for IR (see ``ir.information``).

   .. function:: rqs.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/rqs/Python/test_rqs.py
   :code: python

This example code is available in $GALAHAD/src/rqs/Python/test_rqs.py .
