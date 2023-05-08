LSQP
====

.. module:: galahad.lsqp

The ``lsqp`` package uses an **interior-point trust-region method** to solve a
given **linear or separable convex quadratic program**.
The aim is to minimize the separable quadratic objective function
$$s(x) = f + g^T x + \frac{1}{2} \sum_{j=1}^n w_j^2 (x_j - x_j^0)^2,$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $A$ is a given $m$ by $n$ matrix,  
$g$, $w$ and $x^0$ are vectors, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

In the special case where $w = 0$, $g = 0$ and $f = 0$,
the so-called *analytic center* of the feasible set will be found,
while *linear programming*, or *constrained least distance*, problems
may be solved by picking $w = 0$, or $g = 0$ and $f = 0$, respectively.

See Section 4 of $GALAHAD/doc/lsqp.pdf for additiional details.

The more-modern package ``cqp`` offers similar functionality, and
is often to be preferred.

terminolgy
----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c\;\;\mbox{(1a)}$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,\;\;\mbox{(1b)}$$
the **dual optimality conditions**
$$W^2 ( x - x^0 ) + g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; 
z = z_l + z_u,\;\;\mbox{(2a)}$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 \;\;\mbox{and}\;\; 
z_u \leq 0,\;\;\mbox{(2b)}$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,\;\;\mbox{(3)}$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

method
------

Primal-dual interior point methods iterate towards a point that satisfies 
these optimality conditions by ultimately aiming to satisfy
(1a), (2a) and (3), while ensuring that (1b) and (2b) are
satisfied as strict inequalities at each stage.
Appropriate norms of the amounts by
which (1a), (2a) and (3) fail to be satisfied are known as the
primal and dual infeasibility, and the violation of complementary slackness,
respectively. The fact that (1b) and (2b) are satisfied as strict
inequalities gives such methods their other title, namely
interior-point methods.

When $w \neq 0$ or $g \neq 0$, the method aims at each stage to reduce the
overall violation of (1a), (2a) and (3),
rather than reducing each of the terms individually. Given an estimate
$v = (x, \; c, \; y, \; y^{l}, \; y^{u}, \; z, \; z^{l}, \; z^{u})$
of the primal-dual variables, a correction
$\Delta v = \Delta (x, \; c, \; y, \; y^{l}, \; 
y^{u} ,\;z,\;z^{l} ,\;z^{u} )$
is obtained by solving a suitable linear system of Newton equations for the
nonlinear systems (1a), (2a) and a parameterized perturbation
of (3). An improved estimate $v + \alpha \Delta v$
is then used, where the step-size $\alpha$
is chosen as close to 1.0 as possible while ensuring both that
(1b) and (2b) continue to hold and that the individual components
which make up the complementary slackness
(3) do not deviate too significantly
from their average value. The parameter that controls the perturbation
of (3) is ultimately driven to zero.

The Newton equations are solved  by applying the matrix factorization 
package ``SBLS``, but there are options
to factorize the matrix as a whole (the so-called "augmented system"
approach), to perform a block elimination first (the "Schur-complement"
approach), or to let the method itself decide which of the two
previous options is more appropriate.
The "Schur-complement" approach is usually to be preferred when all the
weights are nonzero or when every variable is bounded (at least one side),
but may be inefficient if any of the columns of $A$ is too dense.

When $w = 0$ and $g = 0$, the method aims instead firstly to find an 
interior primal feasible point, that is to ensure that (1a) is satisfied. 
One this has been achieved, attention is switched to mninizing the
potential function
$$\phi (x,\;c) =
- \sum_{i=1}^{m} \log ( c_{i}  -  c_{i}^{l} )
- \sum_{i=1}^{m} \log ( c_{i}^{u}  -  c_{i} )
- \sum_{j=1}^{n} \log ( x_{j}  -  x_{j}^{l} ) 
- \sum_{j=1}^{n} \log ( x_{j}^{u}  -  x_{j} ) ,$$
while ensuring that (1a) remain satisfied and that 
$x$ and $c$ are strictly interior points for (1b). 
The global minimizer of this minimization problem is known as the
analytic center of the feasible region, and may be viewed as a feasible 
point that is as far from the boundary of the constraints as possible.
Note that terms in the above sumations corresponding to infinite bounds are
ignored, and that equality constraints are treated specially.
Appropriate "primal" Newton corrections are used to generate a sequence
of improving points converging to the analytic center, while the iteration
is stabilized by performing inesearches along these corrections with respect to
$\phi (x,\;c)$.

In order to make the solution as efficient as possible, the 
variables and constraints are reordered internally by the package 
``QPP`` prior to solution. In particular, fixed variables, and 
free (unbounded on both sides) constraints are temporarily removed.

references
----------

The basic algorithm is that of

  Y. Zhang,
  ``On the convergence of a class of infeasible interior-point methods 
  for the horizontal linear complementarity problem''.
  *SIAM J. Optimization* **4(1)** (1994) 208-227,

with a number of enhancements described by

  A. R. Conn, N. I. M. Gould, D. Orban and Ph. L. Toint,
  ``A primal-dual trust-region algorithm for minimizing a non-convex 
  function subject to general inequality and linear equality constraints''.
  *Mathematical Programming **87** (1999) 215-249.


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


functions
---------

   .. function:: lsqp.initialize()

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
          maxit : int
             at most maxit inner iterations are allowed.
          factor : int
             the factorization to be used. Possible values are

             * **0**

               automatic

             * **1**

               Schur-complement factorization

             * **2**

               augmented-system factorization.

          max_col : int
             the maximum number of nonzeros in a column of A which is
             permitted with the Schur-complement factorization.
          indmin : int
             an initial guess as to the integer workspace required by
             SBLS.
          valmin : int
             an initial guess as to the real workspace required by SBLS.
          itref_max : int
             the maximum number of iterative refinements allowed.
          infeas_max : int
             the number of iterations for which the overall
             infeasibility of the problem is not reduced by at least a
             factor ``reduce_infeas`` before the problem is flagged as
             infeasible (see reduce_infeas).
          muzero_fixed : int
             the initial value of the barrier parameter will not be
             changed for the first muzero_fixed iterations.
          restore_problem : int
             indicate whether and how much of the input problem should
             be restored on output. Possible values are

             * **0**

               nothing restored

             * **1**

               scalar and vector parameters

             * **2**

               all parameters.

          indicator_type : int
             specifies the type of indicator function used. Possible
             values are

             * **1**

               primal indicator: constraint active if and only if the
               distance to nearest bound $\f$\leq\f$ ``indicator_p_tol``

             * **2**

               primal-dual indicator: constraint active if and only
               if the  distance to nearest bound $\f$\leq\f$
               ``indicator_tol_pd`` * **size** of  corresponding multiplier

             * **3**

               primal-dual indicator: constraint active if and only
               if the  distance to the nearest bound $\f$\leq\f$
               ``indicator_tol_tapia`` * distance to same bound at
               previous iteration.

          extrapolate : int
             should extrapolation be used to track the central path?
             Possible values

             * **0**

               never

             * **1**

               after the final major iteration

             * **2**

               at each major iteration (unused at present).

          path_history : int
             the maximum number of previous path points to use when
             fitting the data (unused at present).
          path_derivatives : int
             the maximum order of path derivative to use (unused at
             present).
          fit_order : int
             the order of (Puiseux) series to fit to the path data:
             $$\leq$0 to fit all data (unused at present).
          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_p : float
             the required accuracy for the primal infeasibility.
          stop_d : float
             the required accuracy for the dual infeasibility.
          stop_c : float
             the required accuracy for the complementarity.
          prfeas : float
             initial primal variables will not be closer than prfeas
             from their bounds.
          dufeas : float
             initial dual variables will not be closer than dufeas from
             their bounds.
          muzero : float
             the initial value of the barrier parameter. If muzero is
             not positive, it will be reset to an appropriate value.
          reduce_infeas : float
             if the overall infeasibility of the problem is not reduced
             by at least a factor reduce_infeas over ``infeas_max``
             iterations, the problem is flagged as infeasible (see
             infeas_max).
          potential_unbounded : float
             if W=0 and the potential function value is smaller than
             potential_unbounded * number of one-sided bounds, the
             analytic center will be flagged as unbounded.
          pivot_tol : float
             the threshold pivot used by the matrix factorization. See
             the documentation for SBLS for details.
          pivot_tol_for_dependencies : float
             the threshold pivot used by the matrix factorization when
             attempting to detect linearly dependent constraints. See
             the documentation for SBLS for details.
          zero_pivot : float
             any pivots smaller than zero_pivot in absolute value will
             be regarded to zero when attempting to detect linearly
             dependent constraints.
          identical_bounds_tol : float
             any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that
             are closer tha identical_bounds_tol will be reset to the
             average of their values.
          mu_min : float
             start terminal extrapolation when mu reaches mu_min.
          indicator_tol_p : float
             if ``indicator_type`` = 1, a constraint/bound will be
             deemed to be active if and only if the distance to nearest
             bound  $\leq$ ``indicator_p_tol``.
          indicator_tol_pd : float
             if ``indicator_type`` = 2, a constraint/bound will be
             deemed to be active if and only if the distance to nearest
             bound $\leq$ ``indicator_tol_pd`` * size of
             corresponding multiplier.
          indicator_tol_tapia : float
             if ``indicator_type`` = 3, a constraint/bound will be
             deemed to be active if and only if the distance to nearest
             bound $\leq$ ``indicator_tol_tapia`` * distance to
             same bound at previous iteration.
          cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
          clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means
             infinite).
          remove_dependencies : bool
             the equality constraints will be preprocessed to remove
             any linear dependencies if True.
          treat_zero_bounds_as_general : bool
             any problem bound with the value zero will be treated as
             if it were a general value if True.
          just_feasible : bool
             if ``just_feasible`` is True, the algorithm will stop as
             soon as a feasible point is found. Otherwise, the optimal
             solution to the problem will be found.
          getdua : bool
             if ``getdua,`` is True, advanced initial values are
             obtained for the dual variables.
          puiseux : bool
             If extrapolation is to be used, decide between Puiseux and
             Taylor series.
          feasol : bool
             if ``feasol`` is True, the final solution obtained will be
             perturbed so tha variables close to their bounds are moved
             onto these bounds.
          balance_initial_complentarity : bool
             if ``balance_initial_complentarity`` is True, the initial
             complemetarity is required to be balanced.
          use_corrector : bool
             if ``use_corrector,`` a corrector step will be used.
          array_syntax_worse_than_do_loop : bool
             if ``array_syntax_worse_than_do_loop`` is True, f77-style
             do loops will be used rather than f90-style array syntax
             for vector operations.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
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

   .. function:: lsqp.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
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
          total number of entries, in the sparse row-wise storage 
          scheme. It need not be set when the other schemes are used, and in 
          this case can be None.
      options : dict, optional
          dictionary of control options (see ``lsqp.initialize``).


   .. function:: lsqp.solve_qp(n, m, f, g, w, x0, a_ne, A_val, c_l, c_u, x_l, x_u, x, y, z)

      Find a solution to the quadratic program involving the
      separable quadratic objective function $s(x)$.

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
          ``lsqp.load``.
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
      x : ndarray(n)
          holds the initial estimate of the minimizer $x$, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $x=0$, suffices and will be adjusted accordingly.
      y : ndarray(m)
          holds the initial estimate of the Lagrange multipliers $y$
          associated with the general constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $y=0$, suffices and will be adjusted accordingly.
      z : ndarray(n)
          holds the initial estimate of the dual variables $z$
          associated with the simple bound constraints, if known.
          This is not crucial, and if no suitable value is known, then any
          value, such as $z=0$, suffices and will be adjusted accordingly.

      **Returns:**

      x : ndarray(n)
          holds the values of the approximate minimizer $x$ after
          a successful call.
      c : ndarray(m)
          holds the values of the residuals $c(x) = Ax$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the return status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      xstat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] lsqp.information()

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

            * **-5**

              The constraints appear to have no feasible point.

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

            * **-15**

              The Hessian $H$ appears not to be positive definite.

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

            * **-23** 

              An entry from the strict upper triangle of $H$ has been 
              specified.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          iter : int
             the total number of iterations required.
          factorization_status : int
             the return status from the factorization.
          factorization_integer : long
             the total integer workspace required for the factorization.
          factorization_real : long
             the total real workspace required for the factorization.
          nfacts : int
             the total number of factorizations performed.
          nbacts : int
             the total number of "wasted" function evaluations during
             the linesearch.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by LSQP_solve_qp.
          potential : float
             the value of the logarithmic potential function sum
             -log(distance to constraint boundary).
          non_negligible_pivot : float
             the smallest pivot which was not judged to be zero when
             detecting linear dependent constraints.
          feasible : bool
             is the returned "solution" feasible?.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               preprocess : float
                  the CPU time spent preprocessing the problem.
               find_dependent : float
                  the CPU time spent detecting linear dependencies.
               analyse : float
                  the CPU time spent analysing the required matrices prior
                  to factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing the search direction.
               clock_total : float
                  the total clock time spent in the package.
               clock_preprocess : float
                  the clock time spent preprocessing the problem.
               clock_find_dependent : float
                  the clock time spent detecting linear dependencies.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorization.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing the search direction.
          fdc_inform : dict
             inform parameters for FDC (see ``fdc.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).


   .. function:: lsqp.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/lsqp/Python/test_lsqp.py
   :code: python

This example code is available in $GALAHAD/src/lsqp/Python/test_lsqp.py .
