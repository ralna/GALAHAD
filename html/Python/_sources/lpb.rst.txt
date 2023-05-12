LPB
===

.. module:: galahad.lpb

The ``lpb`` package uses a **primal-dual interior-point method** to solve a
given **linear program**.
The aim is to minimize the linear objective function
$$q(x) = f + g^T x$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $A$ is a given $m$ by $n$ matrix,  
$g$ is a vector, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method offers the choice of direct and iterative solution of the key
regularization subproblems, and is most suitable for problems
involving a large number of unknowns $x$.

See Section 4 of $GALAHAD/doc/lpb.pdf for additional details.

terminolgy
----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c\;\;\mbox{(1a)}$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,\;\;\mbox{(1b)}$$
the **dual optimality conditions**
$$g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; 
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

The method aims at each stage to reduce the
overall violation of (1a), (2a) and (3),
rather than reducing each of the terms individually. Given an estimate
$v = (x, \; c, \; y, \; y^{l}, \; y^{u}, \; z, \; z^{l}, \; z^{u})$
of the primal-dual variables, a correction
$\Delta v = \Delta (x, \; c, \; y, \; y^{l}, \; 
y^{u} ,\;z,\;z^{l} ,\;z^{u} )$
is obtained by solving a suitable linear system of Newton equations for the
nonlinear systems (1a), (2a) and a parameterized ``residual
trajectory'' perturbation of (3); residual trajectories
proposed by Zhang (1994) and Zhao and Sun (1999) are possibilities.
An improved estimate $v + \alpha \Delta v$
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

Optionally, the problem may be pre-processed temporarily to eliminate dependent
constraints using the package ``FDC``. This may improve the
performance of the subsequent iteration.

references
----------

The basic algorithm is a generalisation of those of

  Y. Zhang,
  ``On the convergence of a class of infeasible interior-point methods 
  for the horizontal linear complementarity problem''.
  *SIAM J. Optimization* **4(1)** (1994) 208-227,

and 

  G. Zhao and J. Sun,
  ``On the rate of local convergence of high-order infeasible 
  path-following algorithms for $P_*$ linear complementarity problems''.
  *Computational Optimization and Applications* **14(1)* (1999) 293-307,

with many enhancements described by

  N. I. M. Gould, D. Orban and D. P. Robinson,
  ``Trajectory-following methods for large-scale degenerate 
  convex quadratic programming'',
  *Mathematical Programming Computation* **5(2)** (2013) 113-142.


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

   .. function:: lpb.initialize()

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
             specifies the type of indicator function used. Possible values are

             * **1**

               primal indicator: a constraint is active if and only
               if  the distance to its nearest bound <= ``indicator_p_tol``.

             * **2**

               primal-dual indicator: a constraint is active if and only if the
               distance to its nearest bound <= ``indicator_tol_pd`` times the
               size of the corresponding multiplier.

             * **3**

               primal-dual indicator: a constraint is active if and
               only if  the distance to its nearest bound <=
               ``indicator_tol_tapia`` times the distance to same bound at the
               previous iteration.

          arc : int
             which residual trajectory should be used to aim from the
             current iteration to the solution. Possible values are

             * **1**

               the Zhang linear residual trajectory.

             * **2**

               the Zhao-Sun quadratic residual trajectory.

             * **3**

               the Zhang arc ultimately switching to the Zhao-Sun
               residual trajectory.

             * **4**

               the mixed linear-quadratic residual trajectory.

          series_order : int
             the order of (Taylor/Puiseux) series to fit to the path
             data.
          sif_file_device : int
             specifies the unit number to write generated SIF file
             describing the current problem.
          qplib_file_device : int
             specifies the unit number to write generated QPLIB file
             describing the current problem.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          stop_abs_p : float
             the required absolute and relative accuracies for the
             primal infeasibility.
          stop_rel_p : float
             see stop_abs_p.
          stop_abs_d : float
             the required absolute and relative accuracies for the dual
             infeasibility.
          stop_rel_d : float
             see stop_abs_d.
          stop_abs_c : float
             the required absolute and relative accuracies for the
             complementarity.
          stop_rel_c : float
             see stop_abs_c.
          prfeas : float
             initial primal variables will not be closer than prfeas
             from their bound.
          dufeas : float
             initial dual variables will not be closer than dufeas from
             their bounds.
          muzero : float
             the initial value of the barrier parameter. If muzero is
             not positive, it will be reset to an appropriate value.
          tau : float
             the weight attached to primal-dual infeasibility compared
             to complementarity when assessing step acceptance.
          gamma_c : float
             individual complementarities will not be allowed to be
             smaller than gamma_c times the average value.
          gamma_f : float
             the average complementarity will not be allowed to be
             smaller than gamma_f times the primal/dual infeasibility.
          reduce_infeas : float
             if the overall infeasibility of the problem is not reduced
             by at least a factor reduce_infeas over ``infeas_max``
             iterations, the problem is flagged as infeasible (see
             infeas_max).
          obj_unbounded : float
             if the objective function value is smaller than
             obj_unbounded, it will be flagged as unbounded from below.
          potential_unbounded : float
             if W=0 and the potential function value is smaller than
             potential_unbounded * number of one-sided bounds, the
             analytic center will be flagged as unbounded.
          identical_bounds_tol : float
             any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that
             are closer than identical_bounds_tol will be reset to the
             average of their values.
          mu_lunge : float
             start terminal extrapolation when mu reaches mu_lunge.
          indicator_tol_p : float
             if ``indicator_type`` = 1, a constraint/bound will be
             deemed to be active if and only if distance to nearest
             bound <= ``indicator_p_tol``.
          indicator_tol_pd : float
             if ``indicator_type`` = 2, a constraint/bound will be
             deemed to be active if and only if distance to nearest
             bound <= ``indicator_tol_pd`` * size of corresponding
             multiplier.
          indicator_tol_tapia : float
             if ``indicator_type`` = 3, a constraint/bound will be
             deemed to be active if and only if distance to nearest
             bound <= ``indicator_tol_tapia`` * distance to same bound
             at previous iteration.
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
             decide between Puiseux and Taylor series approximations to
             the arc.
          every_order : bool
             try every order of series up to series_order?.
          feasol : bool
             if ``feasol`` is True, the final solution obtained will be
             perturbed so tha variables close to their bounds are moved
             onto these bounds.
          balance_initial_complentarity : bool
             if ``balance_initial_complentarity`` is True, the initial
             complemetarity is required to be balanced.
          crossover : bool
             if ``crossover`` is True, cross over the solution to one
             defined by linearly-independent constraints if possible.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          generate_sif_file : bool
             if ``generate_sif_file`` is True if a SIF file
             describing the current problem is to be generated.
          generate_qplib_file : bool
             if ``generate_qplib_file`` is True if a QPLIB file
             describing the current problem is to be generated.
          sif_file_name : str
             name of generated SIF file containing input problem.
          qplib_file_name : str
             name of generated QPLIB file containing input problem.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          fdc_options : dict
             default control options for FDC (see ``fdc.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          fit_options : dict
             default control options for FIT (see ``fit.initialize``).
          roots_options : dict
             default control options for ROOTS (see ``roots.initialize``).
          cro_options : dict
             default control options for CRO (see ``cro.initialize``).

   .. function:: lpb.load(n, m, A_type, A_ne, A_row, A_col, A_ptr, options=None)

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
          dictionary of control options (see ``lpb.initialize``).

   .. function:: lpb.solve_lp(n, m, f, g, a_ne, A_val, c_l, c_u, x_l, x_u)

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
      a_ne : int
          holds the number of entries in the constraint Jacobian $A$.
      A_val : ndarray(a_ne)
          holds the values of the nonzeros in the constraint Jacobian
          $A$ in the same order as specified in the sparsity pattern in 
          ``lpb.load``.
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
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.

   .. function:: [optional] lpb.information()

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
              its relevant string 'dense', 'coordinate' or 'sparse_by_rows'
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
          threads : int
             the number of threads used.
          obj : float
             the value of the objective function at the best estimate
             of the solution determined by LPB_solve.
          primal_infeasibility : float
             the value of the primal infeasibility.
          dual_infeasibility : float
             the value of the dual infeasibility.
          complementary_slackness : float
             the value of the complementary slackness.
          init_primal_infeasibility : float
             these values at the initial point (needed bg GALAHAD_CLPB).
          init_dual_infeasibility : float
             see init_primal_infeasibility.
          init_complementary_slackness : float
             see init_primal_infeasibility.
          potential : float
             the value of the logarithmic potential function sum
             -log(distance to constraint boundary).
          non_negligible_pivot : float
             the smallest pivot which was not judged to be zero when
             detecting linearly dependent constraints.
          feasible : bool
             is the returned "solution" feasible?.
          checkpointsIter : ndarray(16)
             checkpointsIter(i) records the iteration at which the
             criticality measures first fall below 
             $10^{-i-1}, i = 0, \ldots 15$ (where -1 means not achieved).
          checkpointsTime : ndarray(16)
             checkpointsTime(i) records the CPU time at which the
             criticality measures first fall below 
             $10^{-i-1}, i = 0, \ldots 15$ (where -1 means not achieved).
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
          fit_inform : dict
             return information from FIT (see ``fit.information``).
          roots_inform : dict
             return information from ROOTS (see ``roots.information``).
          cro_inform : dict
             inform parameters for CRO (see ``cro.information``).
          rpd_inform : dict
             inform parameters for RPD (see ``rpd.information``).


   .. function:: lpb.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/lpb/Python/test_lpb.py
   :code: python

This example code is available in $GALAHAD/src/lpb/Python/test_lpb.py .
