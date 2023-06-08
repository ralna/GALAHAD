LLSR
====

.. module:: galahad.llsr

Given a real $m$ by $n$ model matrix $A$, a real $n$ by $n$ symmetric
diagonally-dominant matrix $S$, a real $m$ vector of observations $b$
and a scalar $\Delta>0$, the ``llsr`` package finds a **minimizer of the
regularized linear least-squares objective function**
$$\frac{1}{2} \| A x  - b \|^2_2 + \frac{\sigma}{p}\|x\|_S^p,$$
where the $S$-norm of $x$ is $\|x\|_S = \sqrt{x^T S x}$.
This problem commonly occurs as a subproblem in nonlinear
least-squares calculations.
The matrix $S$ need not be provided in the commonly-occurring
$\ell_2$-regularization case for which $S = I$, the $n$ by $n$
identity matrix.

Factorization of matrices of the form
$$\begin{pmatrix}\lambda S & A^T \\ A & - I\end{pmatrix}\mspace{5em}\mbox{(1)}$$
for a succession of scalars $\lambda$ will be required, so this package is
most suited for the case where such a factorization may be found efficiently.
If this is not the case, the package ``lsrt`` may be preferred.

See Section 4 of $GALAHAD/doc/llsr.pdf for additional details.

method
------

The required solution $x_*$ necessarily satisfies the optimality condition
$A^T A x_* + \lambda_* S x_* = A^T b$,
where $\lambda_* = \sigma \|x_*\|^{p-2}$.

The method is iterative, and proceeds in two phases.
Firstly, lower and upper bounds, $\lambda_L$ and
$\lambda_U$, on $\lambda_*$ are computed
using Gershgorin's theorems and other eigenvalue bounds,
including those that may involve the Cholesky factorization of $S$  The
first phase of the computation proceeds by progressively shrinking  the bound
interval $[\lambda_L,\lambda_U]$ until a value $\lambda$ for which
$\|x(\lambda)\|_{M}  \geq  \sigma \|x(\lambda)\|_S^{p-2}$ is found.
Here $x(\lambda)$ and its companion $y(\lambda)$
are defined to be a solution of
$$(A^T A  + \lambda S)x(\lambda) = A^T b. \mspace{5em}\mbox{(2)}$$
Once the terminating
$\lambda$ from the first phase has been discovered, the second phase
consists of applying Newton or higher-order iterations to the nonlinear
*secular* equation $\lambda = \sigma \|x(\lambda)\|_S^{p-2}$ with
the knowledge that such iterations are both globally and ultimately
rapidly convergent.

The dominant cost is the requirement that we solve a sequence of
linear systems (2). This may be rewritten as
$$\begin{pmatrix}\lambda S & A^T \\ A & - I\end{pmatrix}
\begin{pmatrix}x(\lambda) \\ y(\lambda)\end{pmatrix} =
\begin{pmatrix}A^T b \\ 0\end{pmatrix} \mspace{5em} \mbox{(3)}$$
for some auxiliary vector $y(\lambda)$.
In general a sparse symmetric, indefinite factorization of the
coefficient matrix of (3) is
often preferred to a Cholesky factorization of that of (2).

reference
---------

The method is the obvious adaptation to the linear least-squares
problem of that described in detail in

  H. S. Dollar, N. I. M. Gould and D. P. Robinson.
  ``On solving trust-region and other regularised subproblems in optimization''.
  *Mathematical Programming Computation* **2(1)** (2010) 21--57.

matrix storage
--------------

The **unsymmetric** $m$ by $n$ model matrix $A$ may be presented
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
will hold the value $A_{ij}$ for $0 \leq i \leq m-1$, $0 \leq j \leq n-1
The string A_type = 'dense_by_columns' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $A$,
its row index i, column index j and value $A_{ij}$,
$0 \leq i \leq m-1$,  $0 \leq j \leq n-1$,  are stored as the $l$-th
components of the integer arrays A_row and A_col and real array A_val,
respectively, while the number of nonzeros is recorded as A_ne = $ne$.
The string A_type = 'coordinate'should be specified.

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

The **symmetric** $n$ by $n$ scaing matrix $S$ may also
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).

*Dense* storage format:
The matrix $S$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $S$ is
symmetric, only the lower triangular part (that is the part
$S_{ij}$ for $0 \leq j \leq i \leq n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array S_val
will hold the value $S_{ij}$ (and, by symmetry, $S_{ji}$)
for $0 \leq j \leq i \leq n-1$.
The string S_type = 'dense' should be specified.

*Sparse co-ordinate* storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 \leq l \leq ne-1$, of $S$,
its row index i, column index j and value $S_{ij}$,
$0 \leq j \leq i \leq n-1$,  are stored as the $l$-th
components of the integer arrays S_row and S_col and real array S_val,
respectively, while the number of nonzeros is recorded as
S_ne = $ne$. Note that only the entries in the lower triangle
should be stored.
The string S_type = 'coordinate' should be specified.

*Sparse row-wise* storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $S$ the i-th component of the
integer array S_ptr holds the position of the first entry in this row,
while S_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, and values
$S_{ij}$ of the  entries in the i-th row are stored in components
l = S_ptr(i), ..., S_ptr(i+1)-1 of the
integer array S_col, and real array S_val, respectively. Note that as before
only the entries in the lower triangle should be stored. For sparse matrices,
this scheme almost always requires less storage than its predecessor.
The string S_type = 'sparse_by_rows' should be specified.

*Diagonal* storage format:
If $S$ is diagonal (i.e., $S_{ij} = 0$ for all
$0 \leq i \neq j \leq n-1$) only the diagonals entries
$S_{ii}$, $0 \leq i \leq n-1$ need be stored,
and the first n components of the array S_val may be used for the purpose.
The string S_type = 'diagonal' should be specified.

*Multiples of the identity* storage format:
If $S$ is a multiple of the identity matrix, (i.e., $H = \alpha I$
where $I$ is the n by n identity matrix and $\alpha$ is a scalar),
it suffices to store $\alpha$ as the first component of S_val.
The string S_type = 'scaled_identity' should be specified.

The *identity matrix* format:
If $S$ is the identity matrix, no values need be stored.
The string S_type = 'identity' should be specified. Strictly
this is not required as $S$ will be assumed to be $I$ if it
is not explicitly provided.

The *zero matrix* format:
The same is true if $S$ is the zero matrix, but now
the string S_type = 'zero' or 'none' should be specified.


functions
---------

   .. function:: llsr.initialize()

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
             any printing will start on this iteration.
          stop_print : int
             any printing will stop on this iteration.
          new_a : int
             how much of $A$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1**

               values but not indices have changed

             * **2**

               values and indices have changed.
          new_s : int
             how much of $S$ has changed since the previous call.
             Possible values are

             * **0**

               unchanged

             * **1**

               values but not indices have changed

             * **2**

               values and indices have changed.
          max_factorizations : int
             the maximum number of factorizations (=iterations)
             allowed. -ve implies no limit.
          taylor_max_degree : int
             maximum degree of Taylor approximant allowed (<= 3).
          initial_multiplier : float
             initial estimate of the Lagrange multipler.
          lower : float
             lower and upper bounds on the multiplier, if known.
          upper : float
             see lower.
          stop_normal : float
             stop when $| \|x\| -$ radius $| \leq$ max( stop_normal *
             max( 1, radius ).
          use_initial_multiplier : bool
             ignore initial_multiplier?.
          space_critical : bool
             if space is critical, ensure allocated arrays are no
             bigger than needed.
          deallocate_error_fatal : bool
             exit if any deallocation fails.
          definite_linear_solver : str
             name of the definite linear equation solver employed.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sbls_control : dict
             control parameters for the symmetric factorization and
             related linear solves (see ``sbls.initialize``).
          sls_control : dict
             control parameters for the factorization of $S$ and related
             linear solves (see ``sls.initialize``).
          ir_control : dict
             control parameters for iterative refinement for definite
             system solves (see ``ir.initialize``).

   .. function:: llsr.load(m, n, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      m : int
          holds the number of observations, $m$ (= the number of rows of $A$).
      n : int
          holds the number of variables, $n$ (= the number of columns of $A$).
      A_type : string
          specifies the unsymmetric storage scheme used for the model
          matrix $A$.
          It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
          lower or upper case variants are allowed.
      A_ne : int
          holds the number of entries in $A$ in the sparse co-ordinate storage
          scheme. It need not be set for any of the other two schemes.
      A_row : ndarray(A_ne)
          holds the row indices of $A$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other two schemes, and in this case can be None.
      A_col : ndarray(A_ne)
          holds the column indices of $A$ in either the sparse co-ordinate,
          or the sparse row-wise storage scheme. It need not be set when the
          dense storage scheme is used, and in this case can be None.
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the
          total number of entries, in the sparse row-wise storage
          scheme. It need not be set when the other schemes are used, and in
          this case can be None.
      options : dict, optional
          dictionary of control options (see ``llsr.initialize``).

   .. function:: [optional] llsr.load_scaling(n, S_type, S_ne, S_row, S_col, S_ptr, options=None)

      Import non-trivial regularization-norm scaling data into internal storage
      prior to solution. This is only required if $S$ is not the identity
      matrix $I$.

      **Parameters:**

      n : int
          holds the number of variables, $n$
          (= the number of rows/columns of $S$).
      S_type : string
          specifies the symmetric storage scheme used for the scaling matrix
          $S$. It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal', 'scaled_identity', 'identity', 'zero'  or 'none';
          lower or upper case variants are allowed.
      S_ne : int
          holds the number of entries in the  lower triangular part of
          $S$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other schemes.
      S_row : ndarray(S_ne)
          holds the row indices of the lower triangular part of $S$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other schemes, and in this case can be None.
      S_col : ndarray(S_ne)
          holds the column indices of the  lower triangular part of
          $S$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the other storage schemes
          are used, and in this case can be None.
      S_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $S$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None.
      options : dict, optional
          dictionary of control options (see ``llsr.initialize``).

   .. function:: llsr.solve_problem(m, n, power, weight, A_ne, A_val, b, S_ne, S_val)

      Solve the regularized linear-least-squares problem.

      **Parameters:**

      m : int
          holds the number of observations, $m$.
      n : int
          holds the number of variables, $n$.
      power : float
          holds the regularization power, $\weight \geq 2$.
      weight : float
          holds the strictly positive regularization weight, $\weight$.
      a_ne : int
          holds the number of entries in the model matrix $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in $A$ in the same order as
          specified in the sparsity pattern in ``llsr.load``.
      b : ndarray(m)
          holds the values of the observations $b$.
      s_ne : int, optional
          holds the number of entries in the lower triangular part of
          the scaling matrix $S$ if it is not the identity matrix.
          Otherwise it should be None.
      S_val : ndarray(s_ne), optional
          holds the values of the nonzeros in the lower triangle of $S$ in
          the same order as specified in the sparsity pattern in
          ``llsr.load_scaling`` if needed. Otherwise it should be None.

      **Returns:**

      x : ndarray(n)
          holds the values of the minimizer $x$ after a successful call.

   .. function:: [optional] llsr.information()

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

              The restriction n > 0 or m > 0 or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal', 'scaled_identity',  'identity', 'zero' or 'none'
              has been violated.

            * **-9**

              The analysis phase of the factorization of $K(\lambda)$ failed;
              the return status from the factorization package is given by
              inform['factor_status'].

            * **-10**

              The factorization of $K(\lambda)$ failed; the return status from
              the factorization package is given by inform['factor_status'].

            * **-11**

              The solution of a set of linear equations using factors
              from the factorization package failed; the return status
              from the factorization package is given by
              inform['factor_status'].

            * **-15**

              The Hessian $S$ appears not to be strictly diagonally
              dominant.

            * **-16**

              The problem is so ill-conditioned that further progress
              is impossible.

            * **-17**

              The step is too small to make further progress.

            * **-23**

              An entry from the strict upper triangle of $S$ has been specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          factorizations : int
             the number of factorizations performed.
          len_history : int
             the number of $(\lambda,\|x\|_S,\|Ax-b\|)$ triples in the history.
          multiplier : float
             the Lagrange multiplier corresponding to the regularization term.
          x_norm : float
             the S-norm of x, $\|x\|_S$.
          r_norm : float
             corresponding value of the two-norm of the residual,
             $\|A x(\lambda) - b\|$.
          time : dict
             dictionary containing timing information:
               total : float
                  total CPU time spent in the package.
               assemble : float
                  CPU time assembling $K(\lambda)$ in (1).
               analyse : float
                  CPU time spent analysing $K(\lambda)$.
               factorize : float
                  CPU time spent factorizing $K(\lambda)$.
               solve : float
                  CPU time spent solving linear systems inolving
                  $K(\lambda)$.
               clock_total : float
                  total clock time spent in the package.
               clock_assemble : float
                  clock time assembling $K(\lambda)$.
               clock_analyse : float
                  clock time spent analysing $K(\lambda)$.
               clock_factorize : float
                  clock time spent factorizing $K(\lambda)$.
               clock_solve : float
                  clock time spent solving linear systems inolving
                  $K(\lambda)$.
          history : dict
             dictionary recording the history of the iterates:
               lambda : ndarray(100)
                  the values of $\lambda$ for the first min(100,
                  ``len_history``) iterations.
               x_norm : ndarray(100)
                  the corresponding values of $\|x(\lambda)\|_S$.
               r_norm : ndarray(100)
                  the corresponding values of $\|A x(\lambda) - b\|_2$.
          sbls_inform : dict
             information from the symmetric factorization and related
             linear solves (see ``sbls.information``).
          sls_inform : dict
             information from the factorization of S and related linear
             solves (see ``sls.information``).
          ir_inform : dict
             information from the iterative refinement for definite
             system solves (see ``ir.information``).

   .. function:: llsr.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/llsr/Python/test_llsr.py
   :code: python

This example code is available in $GALAHAD/src/llsr/Python/test_llsr.py .
