EQP
===

.. module:: galahad.eqp

The ``eqp`` package uses an **iterative method** to solve a
given **equality-constrained quadratic program**.
The aim is to minimize the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$ 
or the **shifted-least-distance** objective function
$$s(x) = f + g^T x + \frac{1}{2} \sum_{j=1}^n w_j^2 (x_j - x_j^0)^2,$$ 
subject to the general linear equality constraints
$$A x + c = 0,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric and $m$ by $n$ general matrices,  
$g$, $w$, $x^0$ and $c$ are vectors, and  $f$ is a scalar.
The method is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/eqp.pdf for a brief description of the
method employed and other details.

matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$ may be presented
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
The string A_type = 'coordinate'should be specified.

*Sparse row-wise storage* format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries plus one.
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
while A_ptr(n) holds the total number of entries plus one.
The row indices i, $0 \leq i \leq m-1$, and values $A_{ij}$
of the  nonzero entries in the j-th columnsare stored in components
l = A_ptr(j), $\ldots$, A_ptr(j+1)-1, $0 \leq j \leq n-1$,
of the integer array A_row, and real array A_val, respectively.
As before, for sparse matrices, this scheme almost always requires less
storage than the co-ordinate format.
The string A_type = 'sparse_by_columns' should be specified.

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

   .. function:: eqp.initialize()

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

               1 gives a one-line summary for every iteration.

             * **2**

               gives a summary of the inner iteration for each iteration.

             * **>=3**

               gives increasingly verbose (debugging) output.

          factorization : int
             the factorization to be used. Possible values are 

             * **0**

               automatic 

             * **1**

               Schur-complement factorization 

             * **2**

               augmented-system factorization (obsolete).

          max_col : int
             the maximum number of nonzeros in a column of A which is
             permitted with the Schur-complement factorization
             (obsolete).
          indmin : int
             an initial guess as to the integer workspace required by
             SBLS (obsolete).
          valmin : int
             an initial guess as to the real workspace required by SBLS
             (obsolete).
          len_ulsmin : int
             an initial guess as to the workspace required by ULS
             (obsolete).
          itref_max : int
             the maximum number of iterative refinements allowed
             (obsolete).
          cg_maxit : int
             the maximum number of CG iterations allowed. If cg_maxit <
             0, this number will be reset to the dimension of the
             system + 1.
          preconditioner : int
             the preconditioner to be used for the CG. Possible values are

             * **0**

               automatic.

             * **1**

               no preconditioner, i.e, the identity within full factorization.

             * **2**

               full factorization.

             * **3**

               band within full factorization.

             * **4**

               diagonal using the barrier terms within full
               factorization.  (obsolete)

             * **5**

               optionally supplied diagonal, G = D.

          semi_bandwidth : int
             the semi-bandwidth of a band preconditioner, if
             appropriate (obsolete).
          new_a : int
             how much has A changed since last problem solved: 0 = not
             changed, 1 = values changed, 2 = structure changed.
          new_h : int
             how much has H changed since last problem solved: 0 = not
             changed, 1 = values changed, 2 = structure changed.
          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          pivot_tol : float
             the threshold pivot used by the matrix factorization. See
             the documentation for SBLS for details (obsolete).
          pivot_tol_for_basis : float
             the threshold pivot used by the matrix factorization when
             finding the ba See the documentation for ULS for details
             (obsolete).
          zero_pivot : float
             any pivots smaller than zero_pivot in absolute value will
             be regarded to zero when attempting to detect linearly
             dependent constraints (obsolete).
          inner_fraction_opt : float
             the computed solution which gives at least
             inner_fraction_opt times the optimal value will be found
             (obsolete).
          radius : float
             an upper bound on the permitted step (-ve will be reset to
             an appropriat large value by eqp_solve).
          min_diagonal : float
             diagonal preconditioners will have diagonals no smaller
             than min_diagonal (obsolete).
          max_infeasibility_relative : float
             if the constraints are believed to be rank defficient and
             the residual at a "typical" feasible point is larger than
             max( max_infeasibility_relative * norm A,
             max_infeasibility_absolute ) the problem will be marked as
             infeasible.
          max_infeasibility_absolute : float
             see max_infeasibility_relative.
          inner_stop_relative : float
             the computed solution is considered as an acceptable
             approximation to th minimizer of the problem if the
             gradient of the objective in the preconditioning(inverse)
             norm is less than max( inner_stop_relative * initial
             preconditioning(inverse) gradient norm,
             inner_stop_absolute ).
          inner_stop_absolute : float
             see inner_stop_relative.
          inner_stop_inter : float
             see inner_stop_relative.
          find_basis_by_transpose : bool
             if ``find_basis_by_transpose`` is True, implicit
             factorization precondition will be based on a basis of A
             found by examining A's transpose (obsolete).
          remove_dependencies : bool
             if ``remove_dependencies`` is True, the equality
             constraints will be preprocessed to remove any linear
             dependencies.
          space_critical : bool
             if ``space_critical`` True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          generate_sif_file : bool
             if ``generate_sif_file`` is True, a SIF file
             describing the current problem is to be generated.
          sif_file_name : str
             name of generated SIF file containing input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          fdc_options : dict
             default control options for FDC (see ``fdc.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          gltr_options : dict
             default control options for GLTR (see ``gltr.initialize``).

   .. function:: eqp.load(n, m, H_type, H_ne, H_row, H_col, H_ptr, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
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
      A_type : string
          specifies the unsymmetric storage scheme used for the constraints 
          Jacobian $A$.
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
          total number of entries plus one, in the sparse row-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      options : dict, optional
          dictionary of control options (see ``eqp.initialize``).

   .. function:: eqp.solve_qp(n, m, f, g, h_ne, H_val, a_ne, A_val, c)

      Find a solution to the convex quadratic program involving the
      quadratic objective function $q(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
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
          ``eqp.load``.
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``eqp.load``.
      c : ndarray(m)
          holds the values of the linear term $c$ in the constraints

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          linear constraints.

   .. function:: eqp.solve_sldqp(n, m, f, g, w, x0, a_ne, A_val, c_l, c_u, x_l, x_u)

      Find a solution to the quadratic program involving the
      shifted least-distance objective function $s(x)$.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      w : ndarray(n)
          holds the values of the weights $w$ in the objective function.
      x0 : ndarray(n)
          holds the values of the shifts $x^0$ in the objective function.
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``eqp.load``.
      c_l : ndarray(m)
          holds the values of the lower bounds $c_l$ on the constraints
          The lower bound on any component of $A x$ that is unbounded from 
          below should be set no larger than minus ``options.infinity``.
      c_u : ndarray(m)
          holds the values of the upper bounds $c_l$ on the  constraints
          The upper bound on any component of $A x$ that is unbounded from 
          above should be set no smaller than ``options.infinity``.
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
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.

   .. function:: eqp.resolve_qp(n, m, f, g, c)

      Resolve the convex quadratic program when the data $f$, $g$ and/or $c$
      has changed.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of residuals.
      f : float
          holds the constant term $f$ in the objective function.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      c : ndarray(m)
          holds the values of the linear term $c$ in the constraints

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          linear constraints.

   .. function:: [optional] eqp.information()

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

            * **-5**

              The constraints appear to be inconsistent.

            * **-7**

              The objective function appears to be unbounded from below
              on the feasible set.

            * **-9**

              The analysis phase of a symmetric factorization failed; the 
              return status from the factorization package is given by
              inform['factor_status'].

            * **-10**

              A symmetric factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

            * **-11**

              The solution of a set of linear equations using factors
              from a symmetric factorization package failed; the return
              status from the factorization package is given by
              inform['factor_status'].

            * **-12**

              The analysis phase of an unsymmetric factorization failed; the 
              return status from the factorization package is given by
              inform['factor_status'].

            * **-13**

              An unsymmetric factorization failed; the return status from the
              factorization package is given by inform['factor_status'].

            * **-14**

              The solution of a set of linear equations using factors
              from an unsymmetric factorization package failed; the return
              status from the factorization package is given by
              inform['factor_status'].

            * **-15**

              The computed preconditioner has the wrong inertia and is 
              thus unsuitable.

            * **-16**

              The residuals from the preconditioning step are large, 
              indicating that the factorization may be unsatisfactory.

            * **-25** 

              ``eqp.resolve`` has been called before ``eqp.solve``.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error ocurred.
          cg_iter : int
             the total number of conjugate gradient iterations required.
          cg_iter_inter : int
             see cg_iter.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by QPB_solve.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               find_dependent : float
                  the CPU time spent detecting linear dependencies.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
               solve_inter : float
                  see solve.
               clock_total : float
                  the total clock time spent in the package.
               clock_find_dependent : float
                  the clock time spent detecting linear dependencies.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
          fdc_inform : dict
             inform parameters for FDC (see ``fdc.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).
          gltr_inform : dict
             inform parameters for GLTR (see ``gltr.information``).

   .. function:: eqp.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/eqp/Python/test_eqp.py
   :code: python

This example code is available in $GALAHAD/src/eqp/Python/test_eqp.py .
