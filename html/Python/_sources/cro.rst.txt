CRO
===

.. module:: galahad.cro

The ``cro`` package provides a **crossover** from a
**primal-dual interior-point**
solution to given **convex quadratic program** to a **basic one** in which 
the matrix of defining active constraints/variables is of **full rank**. 
This applies to the problem of minimizing the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x$$ 
subject to the general linear constraints and simple bounds
$$c_l \leq A x \leq c_u \;\;\mbox{and} \;\; x_l \leq x \leq x_u,$$
where $H$ and $A$ are, respectively, given 
$n$ by $n$ symmetric postive-semi-definite and $m$ by $n$ matrices,  
$g$ is a vector, $f$ is a scalar, and any of the components 
of the vectors $c_l$, $c_u$, $x_l$ or $x_u$ may be infinite.
The method is most suitable for problems involving a large number of 
unknowns $x$.

See Section 4 of $GALAHAD/doc/cro.pdf for additional details.

terminolgy
----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,$$
the **dual optimality conditions**
$$H x + g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; z = z_l + z_u,$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 
\;\;\mbox{and}\;\; z_u \leq 0,\;\;\mbox{(1)}$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

method
------

Denote the active constraints by $A_A x = c_A$ and the
active bounds by $I_A x = x_A$. Then any optimal solution satisfies
the linear system
$$\left(\begin{array}{ccc}
H & - A_A^T & - I^T_A \\ A_A & 0 & 0 \\ I_A & 0 & 0 
\end{array}\right) \left(\begin{array}{c} 
x \\ y_A \\ z_A
\end{array}\right) = \left(\begin{array}{c} 
- g \\ c_A \\ x_A 
\end{array}\right), $$
where $y_A$ and $z_A$ are the corresponding active Lagrange multipliers
and dual variables respectively. Consequently the difference between
any two solutions $(\Delta x, \Delta y, \Delta z)$ must satisfy
$$\left(\begin{array}{ccc}
H & - A_A^T & - I^T_A \\ A_A & 0 & 0 \\ I_A & 0 & 0 
\end{array}\right) \left(\begin{array}{c}
\Delta x \\ \Delta y_A \\ \Delta z_A
\end{array}\right) = 0.\;\;\mbox{(2)}$$
Thus there can only be multiple solution if the coefficient matrix $K$
of (2) is singular. The algorithm used in \fullpackagename\
exploits this. The matrix $K$ is checked for singularity
using the package ``ULS``. If $K$ is
non singular, the solution is unique and the solution input by the user
provides a linearly independent active set. Otherwise $K$ is singular,
and partitions $A_A^T = ( A_{AB}^T \;\; A_{AN}^T)$ and
$I_A^T = ( I_{AB}^T \;\; I_{AN}^T)$ are found so that
$$\left(\begin{array}{ccc}
H & - A_{AB}^T & - I^T_{AB} \\ A_{AB} & 0 & 0 \\  I_{AB} & 0 & 0 
\end{array}\right)$$
is non-singular and the *non-basic* constraints $A_{AN}^T$
and $I_{AN}^T$ are linearly dependent on the *basic* ones
$( A_{AB}^T \;\; I_{AB}^T)$. In this case (2) is equivalent to
$$\left(\begin{array}{ccc}
H & - A_{AB}^T & - I^T_{AB} \\ A_{AB} & 0 & 0 \\ I_{AB} & 0 & 0 
\end{array}\right) \left(\begin{array}{c}
\Delta x \\ \Delta y_{AB} \\ \Delta z_{AB} 
\end{array}\right) = \left(\begin{array}{c}
A_{AN}^T \\ 0 \\ 0 
\end{array}\right) \Delta y_{AN} + \left(\begin{array}{c}
I^T_{AN} \\ 0 \\ 0 
\end{array}\right) \Delta z_{AN} .\;\;\mbox{(3)}$$
Thus, starting from the user's $(x, y, z)$
and with a factorization of the coefficient matrix of (3)
found by the package ``SLS``,
the alternative solution $(x + \alpha x, y + \alpha y, z + \alpha z)$,
featuring $(\Delta x, \Delta y_{AB}, \Delta z_{AB})$ from (3)
in which successively one of the components of $\Delta y_{AN}$
and $\Delta z_{AN}$ in turn is non zero, is taken.
The scalar $\alpha$ at each stage
is chosen to be the largest possible that guarantees (1);
this may happen when a non-basic multiplier/dual variable reaches zero,
in which case the corresponding constraint is disregarded, or when this
happens for a basic multiplier/dual variable, in which case this constraint is
exchanged with the non-basic one under consideration and disregarded.
The latter corresponds to changing the basic-non-basic partition
in (3), and subsequent solutions may be found by updating
the factorization of the coefficient matrix in (3)
following the basic-non-basic swap using the package ``SCU``.

matrix storage
--------------

The **unsymmetric** $m$ by $n$ matrix $A$ must be presented
and stored in *sparse row-wise storage* format.
For this, only the nonzero entries are stored, and they are
ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $A$ the i-th component of the
integer array A_ptr holds the position of the first entry in this row,
while A_ptr(m) holds the total number of entries.
The column indices j, $0 \leq j \leq n-1$, and values
$A_{ij}$ of the  nonzero entries in the i-th row are stored in components
l = A_ptr(i), $\ldots$, A_ptr(i+1)-1,  $0 \leq i \leq m-1$,
of the integer array A_col, and real array A_val, respectively.

The **symmetric** $n$ by $n$ matrix $H$ must also be presented 
and stored in *sparse row-wise storage* format. But, crucially, now symmetry
is exploited by only storing values from the *lower triangular* part
(i.e, those entries that lie on or below the leading diagonal).
As before, only the nonzero entries of the matrices are stored.
Only the nonzero entries from the lower triangle are stored, and
these are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries.
The column indices j, $0 \leq j \leq i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively.


functions
---------

   .. function:: cro.initialize()

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

               gives no output.

             * **1**

               a summary of the progress made.

             * **>=2**

               an ever increasing amount of debugging information.

          max_schur_complement : int
             the maximum permitted size of the Schur complement before
             a refactorization is performed.
          infinity : float
             any bound larger than infinity in modulus will be regarded
             as infinite.
          feasibility_tolerance : float
             feasibility tolerance for KKT violation.
          check_io : bool
             if ``check_io`` is True, the input (x,y,z) will be fully
             tested for consistency.
          refine_solution : bool
             if ``refine`` solution is True, attempt to satisfy the KKT
             conditions as accurately as possible.
          space_critical : bool
             if ``space_critical`` is True, every effort will be made
             to use as little space as possible. This may result in
             longer computation time.
          deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
          symmetric_linear_solver : str
             indefinite linear equation solver.
          unsymmetric_linear_solver : str
             unsymmetric linear equation solver.
          prefix : str
            all output lines will be prefixed by the string contained
            in quotes within ``prefix``, e.g. 'word' (note the qutoes)
            will result in the prefix word.
          sls_options : dict
             default control options for SLS (see ``sls.initialize``).
          uls_options : dict
             default control options for ULS (see ``uls.initialize``).
          sbls_options : dict
             default control options for SBLS (see ``sbls.initialize``).
          ir_options : dict
             default control options for IR (see ``ir.initialize``).

   .. function:: cro.crossover_solution(n, m, m_equal, g, H_ne, H_val, H_col, H_ptr, A_ne, A_val, A_col, A_ptr, c_l, c_u, x_l, x_u, x, y, z, c_stat, x_stat, options=None)

      Crossover a primal-dual interior-point solution to a basic one.

      **Parameters:**

      n : int
          holds the number of variables.
      m : int
          holds the number of constraints.
      m_equal : int
          holds the number of equality constraints. These **must** occur 
          first in $A$.
      g : ndarray(n)
          holds the values of the linear term $g$ in the objective function.
      H_ne : int
          holds the number of entries in the lower triangular part of $H$.
      H_val : ndarray(H_ptr(n)-1)
          holds the values of the nonzeros of the lower triangular 
          part of $H$ in the sparse row-wise storage scheme.
      H_col : ndarray(H_ptr(n)-1)
          holds the column indices of the nonzeros of the lower triangular 
          part of $H$ in the sparse row-wise storage scheme.
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries.
      A_ne : int
          holds the number of entries in $A$.
      A_val : ndarray(A_ptr(m)-1)
          holds the values of the nonzeros of $A$ in the sparse row-wise
          storage scheme.
      A_col : ndarray(A_ptr(m)-1)
          holds the column indices  of the nonzeros of $A$ in the sparse 
          row-wise storage scheme.
      A_ptr : ndarray(m+1)
          holds the starting position of each row of $A$, as well as the 
          total number of entries.
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
          holds the values of the approximate minimizer $x$.
      y : ndarray(m)
          holds the values of the Lagrange multipliers associated with the 
          general linear constraints.
      z : ndarray(n)
          holds the values of the dual variables associated with the 
          simple bound constraints.
      c_stat : ndarray(m)
          holds the input status for each constraint. The i-th component will 
          be negative if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound, positive if it lies on its upper bound, and 
          zero if it lies between bounds.
      x_stat : ndarray(n)
          holds the input status for each variable. The i-th component will be
          negative if the $i$-th variable lies on its lower bound, 
          positive if it lies on its upper bound, and zero if it lies
          between bounds.
      options : dict, optional
          dictionary of control options (see ``cro.initialize``).

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
          be -1 if the value of the $i$-th constraint $(Ax)_i$) lies on 
          its lower bound and the constraint is basic and active, -2 
          if it is non-basic and active, 1 if it lies on its upper bound 
          and is active, 2 if it non-basic and active, and
          0 if it lies between bounds.
      x_stat : ndarray(n)
          holds the return status for each variable. The i-th component will 
          be -1 if the $i$-th variable lies on 
          its lower bound and the variable is basic and active, -2 
          if it is non-basic and active, 1 if it lies on its upper bound 
          and is active, 2 if it non-basic and active, and
          0 if it lies between bounds.
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

              The restriction n > 0 or m > 0 or 0 <= m_equal <= m 
              has been violated.

            * **-4**

              The bound constraints are inconsistent.

            * **-5**

              The constraints appear to have no feasible point.

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

            * **-16**

              The resuduals are large, the factorization may be unsatisfactory.

          alloc_status : int
             the status of the last attempted allocation/deallocation.
          bad_alloc : str
             the name of the array for which an allocation/deallocation
             error occurred.
          dependent : int
             the number of dependent active constraints.
          time : dict
             dictionary containing timing information:
               total : float
                  the total CPU time spent in the package.
               analyse : float
                  the CPU time spent reordering the matrix prior to
                  factorization.
               factorize : float
                  the CPU time spent factorizing the required matrices.
               solve : float
                  the CPU time spent computing corrections.
               clock_total : float
                  the total clock time spent in the package.
               clock_analyse : float
                  the clock time spent analysing the required matrices prior
                  to factorizat.
               clock_factorize : float
                  the clock time spent factorizing the required matrices.
               clock_solve : float
                  the clock time spent computing corrections.
          sls_inform : dict
             inform parameters for SLS (see ``sls.information``).
          uls_inform : dict
             inform parameters for ULS (see ``uls.information``).
          sbls_inform : dict
             inform parameters for SBLS (see ``sbls.information``).
          scu_status : int
             status value for SCU (see ``scu.status``).
          scu_inform : dict
             inform parameters for SCU (see ``scu.information``).
          ir_inform : dict
             return information from IR (see ``ir.information``).


   .. function:: cro.terminate()

     Deallocate all internal private storage.

example code
------------

.. include:: ../../src/cro/Python/test_cro.py
   :code: python

This example code is available in $GALAHAD/src/cro/Python/test_cro.py .
