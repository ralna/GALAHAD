BGO
===

.. module:: galahad.bgo

The ``bgo`` package uses a multi-start trust-region method to find an
approximation to the global minimizer of a differentiable objective
function $f(x)$ of n variables $x$, subject to simple
bounds $x^l <= x <= x^u$ on the variables. Here, any of the
components of the vectors of bounds $x^l$ and $x^u$
may be infinite. The method offers the choice of direct and
iterative solution of the key trust-region subproblems, and
is suitable for large problems. First derivatives are required,
and if second derivatives can be calculated, they will be exploited -
if the product of second derivatives with a vector may be found but
not the derivatives themselves, that may also be exploited.

The package offers both random multi-start and local-minimize-and-probe
methods to try to locate the global minimizer. There are no theoretical
guarantees unless the sampling is huge, and realistically the success
of the methods decreases as the dimension and nonconvexity increase.

See Section 4 of $GALAHAD/doc/bgo.pdf for additional details.

method
------

A choice of two methods is available. 
In the first, local-minimization-and-probe, approach, local minimization
and univariate global minimization are intermixed. Given a current 
champion $x^S_k$, a local minimizer $x_k$ of $f(x)$ within the
feasible box $x^l \leq x \leq x^u$ is found using ``TRB``.
Thereafter $m$ random directions $p$ are generated, and univariate
local minimizer of $f(x_k + \alpha p)$ as a function of the scalar
$\alpha$ along each $p$ within the interval $[\alpha^L,\alpha^u]$,
where $\alpha^L$ and $\alpha^u$ are the smallest and largest $\alpha$ 
for which $x^l \leq x_k + \alpha p \leq x^u$, 
is performed using ``UGO``. The point $x_k + \alpha p$
that gives the smallest value of $f$ is then selected as the new champion
$x^S_{k+1}$.

The random directions $p$ are chosen in one of three ways. The simplest is
to select the components as
$$p_i = \mbox{pseudo random $\in$} \left\{ 
\begin{array}{rl} 
\mbox{[-1,1]} & \mbox{if} \;\; x^l_i < x_{k,i} < x^u_i \\
\mbox{[0,1]} & \mbox{if} \;\; x_{k,i}  = x^l_i \\
\mbox{[-1,0]} & \mbox{if} \;\;  x_{k,i}  = x^u_i
\end{array}
\right.$$
for each $1 \leq i \leq n$. An alternative is to 
pick $p$ by partitioning each dimension of the feasible "hypercube" box 
into $m$ equal segments, and then selecting sub-boxes randomly within 
this hypercube using Latin hypercube sampling via ``LHS``.
Each components of $p$ is then selected in its sub-box, either uniformly
or pseudo randomly.

The other, random-multi-start, method provided selects $m$ starting points 
at random, either componentwise pseudo randomly in the feasible box, 
or by partitioning each component into $m$ equal segments, assigning each to
a sub-box using Latin hypercube sampling, and finally choosing the
values either uniformly or pseudo randomly. Local minimizers within the
feasible box are then computed by ``TRB``, and
the best is assigned as the current champion. This process is then
repeated until evaluation limits are achieved.

If $n=1$, ``UGO`` is called directly.

We reiterate that there are no theoretical guarantees unless the sampling 
is huge, and realistically the success of the methods decreases as the 
dimension and nonconvexity increase. Thus the methods used should best
be viewed as heuristics.

references
----------

The generic bound-constrained trust-region method is described in detail in

  A. R. Conn, N. I. M. Gould and Ph. L. Toint,
  Trust-region methods.
  SIAM/MPS Series on Optimization (2000),

the univariate global minimization method employed is an extension of that
due to

  D. Lera and Ya. D. Sergeyev,
  ``Acceleration of univariate global optimization algorithms working with
  Lipschitz functions and Lipschitz first derivatives''
  *SIAM J. Optimization* **23(1)** (2013) 508–529,

while the Latin-hypercube sampling method employed is that of

  B. Beachkofski and R. Grandhi,
  ``Improved Distributed Hypercube Sampling'',
  43rd AIAA structures, structural dynamics, and materials conference,
  (2002) 2002-1274.

matrix storage
--------------

The symmetric $n$ by $n$ matrix $H = \nabla^2_{xx}f$ may
be presented and stored in a variety of formats. But crucially symmetry
is exploited by only storing values from the lower triangular part
(i.e, those entries that lie on or below the leading diagonal).

Dense storage format:
The matrix $H$ is stored as a compact  dense matrix by rows, that
is, the values of the entries of each row in turn are stored in order
within an appropriate real one-dimensional array. Since $H$ is
symmetric, only the lower triangular part (that is the part
$H_{ij}$ for $0 <= j <= i <= n-1$) need be held.
In this case the lower triangle should be stored by rows, that is
component $i * i / 2 + j$  of the storage array H_val
will hold the value $H_{ij}$ (and, by symmetry, $H_{ji}$)
for $0 <= j <= i <= n-1$.

Sparse co-ordinate storage format:
Only the nonzero entries of the matrices are stored.
For the $l$-th entry, $0 <= l <= ne-1$, of $H$,
its row index i, column index j and value $H_{ij}$,
$0 <= j <= i <= n-1$,  are stored as the $l$-th
components of the integer arrays H_row and H_col and real array H_val,
respectively, while the number of nonzeros is recorded as
H_ne = $ne$. Note that only the entries in the lower triangle
should be stored.

Sparse row-wise storage format:
Again only the nonzero entries are stored, but this time
they are ordered so that those in row i appear directly before those
in row i+1. For the i-th row of $H$ the i-th component of the
integer array H_ptr holds the position of the first entry in this row,
while H_ptr(n) holds the total number of entries.
The column indices j, $0 <= j <= i$, and values
$H_{ij}$ of the  entries in the i-th row are stored in components
l = H_ptr(i), ..., H_ptr(i+1)-1 of the
integer array H_col, and real array H_val, respectively. Note that
as before only the entries in the lower triangle should be stored. For
sparse matrices, this scheme almost always requires less storage than
its predecessor.

functions
---------

   .. function:: bgo.initialize()

      Set default option values and initialize private data

      **Returns:**

      options : dict
          dictionary containing default control options:
           error : int
             error and warning diagnostics occur on stream error.
           out : int
             general output occurs on stream out.
           print_level : int
             the level of output required. Possible values are:

             * **<= 0**

               no output

             * **1**

               a one-line summary for every improvement

             * **2**

               a summary of each iteration

             * **>= 3**

               increasingly verbose (debugging) output.

           attempts_max : int
             the maximum number of random searches from the best point
             found so far.
           max_evals : int
             the maximum number of function evaluations made.
           sampling_strategy : int
             sampling strategy used. Possible values are

             * **1**

               uniformly spread

             * **2**

               Latin hypercube sampling

             * **3**

               uniformly spread within a Latin hypercube.

           hypercube_discretization : int
             hyper-cube discretization (for sampling stategies 2 and 3).
           alive_unit : int
             removal of the file alive_file from unit alive_unit
             terminates execution.
           alive_file : str
             see alive_unit.
           infinity : float
             any bound larger than infinity in modulus will be regarded as
             infinite.
           obj_unbounded : float
             the smallest value the objective function may take before the
             problem is marked as unbounded.
           cpu_time_limit : float
             the maximum CPU time allowed (-ve means infinite).
           clock_time_limit : float
             the maximum elapsed clock time allowed (-ve means infinite).
           random_multistart : bool
             perform random-multistart as opposed to local minimize and
             probe.
           hessian_available : bool
             is the Hessian matrix of second derivatives available or is
             access only via matrix-vector products?.
           space_critical : bool
             if ``space_critical`` is True, every effort will be made to
             use as little space as possible. This may result in longer
             computation time.
           deallocate_error_fatal : bool
             if ``deallocate_error_fatal`` is True, any array/pointer
             deallocation error will terminate execution. Otherwise,
             computation will continue.
           prefix : str
             all output lines will be prefixed by the string contained
             in quotes within ``prefix``, e.g. 'word' (note the qutoes)
             will result in the prefix word.
           ugo_options : dict
             default control options for UGO (see ``ugo.initialize``).
           lhs_options : dict
             default control options for LHS (see ``lhs.initialize``).
           trb_options : dict
             default control options for TRB (see ``trb.initialize``).

   .. function:: bgo.load(n, x_l, x_u, H_type, H_ne, H_row, H_col, H_ptr, options=None)

      Import problem data into internal storage prior to solution.

      **Parameters:**

      n : int
          holds the number of variables.
      x_l : ndarray(n)
          holds the values $x^l$ of the lower bounds on the
          optimization variables $x$.
      x_u : ndarray(n)
          holds the values $x^u$ of the upper bounds on the
          optimization variables $x$.
      H_type : string
          specifies the symmetric storage scheme used for the Hessian.
          It should be one of 'coordinate', 'sparse_by_rows', 'dense',
          'diagonal' or 'absent', the latter if access to the Hessian
          is via matrix-vector products; lower or upper case variants
          are allowed.
      H_ne : int
          holds the number of entries in the  lower triangular part of
          $H$ in the sparse co-ordinate storage scheme. It need
          not be set for any of the other three schemes.
      H_row : ndarray(H_ne)
          holds the row indices of the lower triangular part of $H$
          in the sparse co-ordinate storage scheme. It need not be set for
          any of the other three schemes, and in this case can be None
      H_col : ndarray(H_ne)
          holds the column indices of the  lower triangular part of
          $H$ in either the sparse co-ordinate, or the sparse row-wise
          storage scheme. It need not be set when the dense or diagonal
          storage schemes are used, and in this case can be None
      H_ptr : ndarray(n+1)
          holds the starting position of each row of the lower triangular
          part of $H$, as well as the total number of entries,
          in the sparse row-wise storage scheme. It need not be set when the
          other schemes are used, and in this case can be None
      options : dict, optional
          dictionary of control options (see ``bgo.initialize``).

   .. function:: bgo.solve(n, H_ne, x, g, eval_f, eval_g, eval_h)

      Find an approximation to the global minimizer of a given function
      subject to simple bounds on the variables using a multistart
      trust-region method.

      **Parameters:**

      n : int
          holds the number of variables.
      H_ne : int
          holds the number of entries in the lower triangular part of $H$.
      x : ndarray(n)
          holds the values of optimization variables $x$.
      eval_f : callable
          a user-defined function that must have the signature:

           ``f = eval_f(x)``

          The value of the objective function $f(x)$
          evaluated at $x$ must be assigned to ``f``.
      eval_g : callable
          a user-defined function that must have the signature:

           ``g = eval_g(x)``

          The components of the gradient $\nabla f(x)$ of the
          objective function evaluated at $x$ must be assigned to ``g``.
      eval_h : callable
          a user-defined function that must have the signature:

           ``h = eval_h(x)``

          The components of the nonzeros in the lower triangle of the Hessian
          $\nabla^2 f(x)$ of the objective function evaluated at
          $x$ must be assigned to ``h`` in the same order as specified
          in the sparsity pattern in ``bgo.load``.

      **Returns:**

      x : ndarray(n)
          holds the value of the approximate global minimizer $x$ after
          a successful call.
      g : ndarray(n)
          holds the gradient $\nabla f(x)$ of the objective function.

   .. function:: [optional] bgo.information()

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

              The restriction n > 0 or requirement that type contains
              its relevant string 'dense', 'coordinate', 'sparse_by_rows',
              'diagonal' or 'absent' has been violated.

            * **-7**

              The objective function appears to be unbounded from below.

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
              options['maxit'] is too small, but may also be symptomatic
              of a badly scaled problem.

            * **-19**

              The CPU time limit has been reached. This may happen if
              options['cpu_time_limit'] is too small, but may also be
              symptomatic of a badly scaled problem.

            * **-82**

              The user has forced termination of the solver by removing
              the file named options['alive_file'] from unit
              options['alive_unit'].

          alloc_status : int
            the status of the last attempted allocation/deallocation.
          bad_alloc : str
            the name of the array for which an allocation/deallocation
            error occurred.
          f_eval : int
            the total number of evaluations of the objective function.
          g_eval : int
            the total number of evaluations of the gradient of the
            objective function.
          h_eval : int
            the total number of evaluations of the Hessian of the
            objective function.
          obj : float
            the value of the objective function at the best estimate of
            the solution determined by ``bgo.solve``.
          norm_pg : float
            the norm of the projected gradient of the objective function
            at the best estimate of the solution determined by ``bgo.solve``.
          time : dict
            dictionary containing timing information:
             total : float
               the total CPU time spent in the package.
             univariate_global : float
               the CPU time spent performing univariate global optimization.
             multivariate_local : float
               the CPU time spent performing multivariate local optimization.
             clock_total : float
               the total clock time spent in the package.
             clock_univariate_global : float
               the clock time spent performing univariate global
               optimization.
             clock_multivariate_local : float
               the clock time spent performing multivariate local
               optimization.
          ugo_inform : dict
            inform parameters for UGO (see ``ugo.information``).
          lhs_inform : dict
            inform parameters for LHS (see ``lhs.information``).
          trb_inform : dict
            inform parameters for TRB (see ``trb.information``).

   .. function:: bgo.terminate()

      Deallocate all internal private storage.

example code
------------

.. include:: ../../src/bgo/Python/test_bgo.py
   :code: python

This example code is available in $GALAHAD/src/bgo/Python/test_bgo.py .
