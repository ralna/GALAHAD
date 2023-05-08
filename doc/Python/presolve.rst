PRESOLVE
========

.. module:: galahad.presolve

The ``presolve`` package **transforms** linear and quadratic programming data 
so that the resulting problem is easier to solve.
This reduced problem may then be passed to an
appropriate solver.  Once the reduced problem has been solved, it is
then a restored to recover the solution for the original formulation.

The package applies presolving techniques to **linear programs**, whose
aim is to minimize the linear objective function
$$\ell(x) = f + g^T x,$$
or **quadratic programs**, which target the quadratic objective function
$$q(x) = f + g^T x + \frac{1}{2} x^T H x,$$
subject to the general linear constraints
$$c_i^l  \leq  a_i^{T}x  \leq  c_i^u, \;\;\; i = 1, \ldots , m,$$
and simple bounds
$$x_j^l  \leq  x_j^{ } \leq  x_j^u , \;\;\; j = 1, \ldots , n,$$
where the scalar $f$, the $n$-dimensional vectors $g$, $x^l$ and
$x^u$, the $m$-dimensional vectors $c^l$ and $c^u$,
the $n \times n$ symmetric matrix $H$ and the $m \times n$ matrix $A$
(whose rows are the vectors $a_i^T$) are given. 
Furthermore, bounds on the Lagrange multipliers $y$ associated with
the general linear constraints and on the dual variables $z$ associated
with the simple bound constraints
$$y_i^l  \leq  y_i  \leq  y_i^u, \;\;\;  i = 1, \ldots , m,$$
and
$$z_i^l  \leq  z_i  \leq  z_i^u, \;\;\;  i = 1, \ldots , n,$$
are also provided, where the $m$-dimensional vectors $y^l$ and
$y^u$, as well as the $n$-dimensional vectors $x^l$ and $x^u$
are given.  Any component of $c^l$, $c^u$, $x^l$, $x^u$,
$y^l$, $y^u$, $z^l$ or $z^u$ may be infinite.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/presolve.pdf for additional details.

method
------

The required solution $x$ of the problem necessarily satisfies
the **primal optimality** conditions
$$A x = c$$
and
$$c^l \leq c \leq c^u, \;\; x^l \leq x \leq x^u,$$
the **dual optimality** conditions
$$H x + g =  A^{T} y + z, \;\;  
y = y^l + y^u \;\; \mbox{and} \;\; z = z^l + z^u,$$
and
$$y^l \geq 0 , \;\; y^u \leq 0 , \;\;  
z^l \geq 0 \;\; \mbox{and} \;\; z^u \leq 0,$$
and the **complementary slackness** conditions
$$
( A x - c^l )^{T} y^l = 0, \;\; ( A x - c^u )^{T} y^u = 0, \;\;
(x -x^l )^{T} z^l = 0 \;\;\mbox{and}\;\; (x -x^u )^{T} z^u = 0,
$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold componentwise.

The purpose of presolving is to exploit these equations in order to reduce the
problem to the standard form defined as follows:

* $*$ The variables are ordered so that their bounds appear in the order

  .. list-table:: variable bound order
     :widths: 20 10 10 10 10 10
     :header-rows: 0

     * - free                
       -       
       -        
       - $x$ 
       -        
       -
     * - non-negativity      
       - 0   
       - $\leq$ 
       - $x$ 
       -        
       -
     * - lower               
       - $x^l$ 
       - $\leq$ 
       - $x$ 
       -        
       -
     * - range               
       - $x^l$ 
       - $\leq$ 
       - $x$ 
       - $\leq$ 
       - $x^u$
     * - upper               
       -       
       -        
       - $x$ 
       - $\leq$ 
       - $x^u$
     * - non-positivity      
       -       
       -        
       - $x$ 
       - $\leq$ 
       - 0

  Fixed variables are removed. Within each category, the variables
  are further ordered so that those with non-zero diagonal Hessian
  entries occur before the remainder.

* $*$ The constraints are ordered so that their bounds appear in the order

  .. list-table:: constraint bound order
     :widths: 20 10 10 10 10 10
     :header-rows: 0

     * - non-negativity      
       - 0   
       - $\leq$ 
       - $A x$ 
       -        
       -
     * - equality            
       - $c^l$ 
       - $=$  
       - $A x$ 
       -        
       -
     * - lower               
       - $c^l$ 
       - $\leq$ 
       - $A x$ 
       -        
       -
     * - range               
       - $c^l$ 
       - $\leq$ 
       - $A x$ 
       - $\leq$ 
       - $c^u$
     * - upper               
       -       
       -        
       - $A x$ 
       - $\leq$ 
       - $c^u$
     * - non-positivity      
       -       
       -        
       - $A x$ 
       - $\leq$ 
       - 0

  Free constraints are removed.

* $*$ In addition, constraints may be removed or bounds tightened, to reduce the
  size of the feasible region or simplify the problem if this is possible, and
  bounds may be tightened on the dual variables and the multipliers
  associated  with the problem.

The presolving algorithm proceeds by applying a (potentially long) series of
simple transformations to the problem, each transformation introducing a
further simplification of the problem. These involve the removal of empty and
singleton rows, the removal of redundant and forcing primal constraints, the
tightening of primal and dual bounds, the exploitation of linear singleton,
linear doubleton and linearly unconstrained columns, the merging dependent
variables, row sparsification and split equalities. Transformations are
applied in successive passes, each pass involving the following actions:

* $*$ remove empty and singletons rows,

* $*$ try to eliminate variables that are linearly unconstrained,

* $*$ attempt to exploit the presence of linear singleton columns,

* $*$ attempt to exploit the presence of linear doubleton columns,

* $*$ complete the analysis of the dual constraints,

* $*$ remove empty and singletons rows,

* $*$ possibly remove dependent variables,

* $*$ analyze the primal constraints,

* $*$ try to make $A$ sparser by combining its rows,

* $*$ check the current status of the variables, dual variables
  and multipliers.

All these transformations are applied to the structure of the original
problem, which is only permuted to standard form after all transformations are
completed. **Note that the Hessian and Jacobian of the resulting reduced
problem are always stored in sparse row-wise format.** The reduced problem is
then solved by a quadratic or linear programming solver, thus ensuring
sufficiently small primal-dual feasibility and complementarity. Finally, the
solution of the simplified problem is re-translated in the
variables/constraints/format of the original problem formulation by a
*restoration* phase.

If the number of problem transformations exceeds 
``options['transf_buffer_size']``,  the transformation buffer size,
then they are saved in a "history" file, whose
name may be chosen by specifying the ``options['transf_file_name']``
parameter.  When this is the case, this file
is subsequently reread by ``presolve_restore``. It must not be altered
by the user.

At the overall level, the presolving process follows one of the two
sequences:

$\boxed{\mbox{initialize}}$
$\rightarrow$ $\bigg[$ $\boxed{\mbox{apply transformations}}$
$\rightarrow$ (solve problem)
$\rightarrow$ $\boxed{\mbox{restore}}$  $\bigg]$
$\rightarrow$ $\boxed{\mbox{terminate}}$

or

$\boxed{\mbox{initialize}}$
$\rightarrow$ $\bigg[$ $\boxed{\mbox{read specfile}}$
$\rightarrow$ $\boxed{\mbox{apply transformations}}$
$\rightarrow$ (solve problem)
$\rightarrow$ $\boxed{\mbox{restore}}$  $\bigg]$
$\rightarrow$ $\boxed{\mbox{terminate}}$

where the procedure's control parameter may be modified by reading the
specfile, and where (solve problem) indicates
that the reduced problem is solved. Each of the "boxed" steps in these
sequences corresponds to calling a specific function in the package.
In the above diagrams, brackated subsequence of
steps means that they can be repeated with problem having the same
structure. The value of the ``new_problem_structure`` argument must be true
on entry to ``presolve_apply`` on the
first time it is used in this repeated subsequence. Such a subsequence must be
terminated by a call to ``presolve_terminate`` before presolving is
applied to a problem with a different structure.

Note that the values of the multipliers and dual variables (and thus of
their respective bounds) depend on the functional form assumed for the
Lagrangian function associated with the problem.  This form is given by
$$L( x, y, z ) = q( x ) - {\tt y_{sign}} * y^T ( A x - c ) 
- {\tt z_{sign}} * z,$$
(considering only active constraints $A x = c$), where the parameters
${\tt y_{sign}}$ and ${\tt z_{sign}}$ are +1 or -1 and can be chosen 
by the user using ``options['y_sign']`` and ``options['z_sign']``.
Thus, if ${\tt y_{sign} = +1}$, the multipliers associated to active constraints
originally posed as inequalities are non-negative if the inequality is a lower
bound and non-positive if it is an upper bound. Obvioulsy they are not
constrained in sign for constraints originally posed as equalities. These
sign conventions are reversed if ${\tt y_{sign} = -1}$.
Similarly, if ${\tt z_{sign} = +1}$, the dual variables associated to active
bounds are non-negative if the original bound is an lower bound, non-positive
if it is an upper bound, or unconstrained in sign if the variables is fixed;
and this convention is reversed in ${\tt z_{sign} = -1}$. The values of 
${\tt z_{sign}}$ and ${\tt y_{sign}}$ may be chosen by setting the corresponding
components of the ``options` to 1 or -1.

references
----------

The algorithm is described in more detail in

  N. I. M. Gould and Ph. L. Toint,
  ``Presolving for quadratic programming''.
  *Mathematical Programming* **100(1)** (2004) 95--132.

functions
---------

   .. function:: presolve.initialize()

      Set default option values and initialize private data.

      **Returns:**

      options : dict
        dictionary containing default control options:
          termination : int
             Determines the strategy for terminating the presolve
             analysis. Possible values are:

             * **1** 

               presolving is continued as long as one of  the sizes
               of the problem (n, m, a_ne, or h_ne) is being reduced;

             * **2** 

               presolving is continued as long as problem
               transformations remain possible. NOTE: the maximum number
               of analysis passes (control.max_nbr_passes) and the
               maximum number of problem transformations
               (control.max_nbr_transforms) set an upper limit on the
               presolving effort irrespective of the choice of
               control.termination. The only effect of this latter
               parameter is to allow for early termination.
          max_nbr_transforms : int
             The maximum number of problem transformations, cumulated
             over all calls to \p presolve.
          max_nbr_passes : int
             The maximum number of analysis passes for problem analysis
             during a single call of \p presolve_transform_problem.
          c_accuracy : float
             The relative accuracy at which the general linear
             constraints are satisfied at the exit of the solver. Note
             that this value is not used before the restoration of the
             problem.
          z_accuracy : float
             The relative accuracy at which the dual feasibility
             constraints are satisfied at the exit of the solver. Note
             that this value is not used before the restoration of the
             problem.
          infinity : float
             The value beyond which a number is deemed equal to plus
             infinity (minus infinity being defined as its opposite).
          out : int
             The unit number associated with the device used for
             printout.
          errout : int
             The unit number associated with the device used for error
             ouput.
          print_level : int
             The level of printout requested by the user. Can take the
             values:

             * **<=0** 

               no printout is produced

             * **1** 

               only reports the major steps in the analysis

             * **2** 

               reports the identity of each problem  transformation

             * **3** 

               reports more details

             * **4** 

               reports lots of information.

             * **>=5** 

               reports a completely silly amount of information.
          dual_transformations : bool
             True if dual transformations of the problem are allowed.
             Note that this implies that the reduced problem is solved
             accurately (for the dual feasibility condition to hold) as
             to be able to restore the problem to the original
             constraints and variables. False prevents dual
             transformations to be applied, thus allowing for inexact
             solution of the reduced problem. The setting of this
             control parameter overides that of get_z, get_z_bounds,
             get_y, get_y_bounds, dual_constraints_freq,
             singleton_columns_freq, doubleton_columns_freq,
             z_accuracy, check_dual_feasibility.
          redundant_xc : bool
             True if the redundant variables and constraints (i.e.
             variables that do not appear in the objective function and
             appear with a consistent sign in the constraints) are to
             be removed with their associated constraints before other
             transformations are attempted.
          primal_constraints_freq : int
             The frequency of primal constraints analysis in terms of
             presolving passes. A value of j = 2 indicates that primal
             constraints are analyzed every 2 presolving passes. A zero
             value indicates that they are never analyzed.
          dual_constraints_freq : int
             The frequency of dual constraints analysis in terms of
             presolving passes. A value of j = 2 indicates that dual
             constraints are analyzed every 2 presolving passes. A zero
             value indicates that they are never analyzed.
          singleton_columns_freq : int
             The frequency of singleton column analysis in terms of
             presolving passes. A value of j = 2 indicates that
             singleton columns are analyzed every 2 presolving passes.
             A zero value indicates that they are never analyzed.
          doubleton_columns_freq : int
             The frequency of doubleton column analysis in terms of
             presolving passes. A value of j indicates that doubleton
             columns are analyzed every 2 presolving passes. A zero
             value indicates that they are never analyzed.
          unc_variables_freq : int
             The frequency of the attempts to fix linearly
             unconstrained variables, expressed in terms of presolving
             passes. A value of j = 2 indicates that attempts are made
             every 2 presolving passes. A zero value indicates that no
             attempt is ever made.
          dependent_variables_freq : int
             The frequency of search for dependent variables in terms
             of presolving passes. A value of j = 2 indicates that
             dependent variables are searched for every 2 presolving
             passes. A zero value indicates that they are never
             searched for.
          sparsify_rows_freq : int
             The frequency of the attempts to make A sparser in terms
             of presolving passes. A value of j = 2 indicates that
             attempts are made every 2 presolving passes. A zero value
             indicates that no attempt is ever made.
          max_fill : int
             The maximum percentage of fill in each row of A. Note that
             this is a row-wise measure: globally fill never exceeds
             the storage initially used for A, no matter how large
             control.max_fill is chosen. If max_fill is negative, no
             limit is put on row fill.
          transf_file_nbr : int
             The unit number to be associated with the file(s) used for
             saving problem transformations on a disk file.
          transf_buffer_size : int
             The number of transformations that can be kept in memory
             at once (that is without being saved on a disk file).
          transf_file_status : int
             The exit status of the file where problem transformations
             are saved:

             * **0** 

               the file is not deleted after program termination

             * **1** 

               the file is not deleted after program termination.
          transf_file_name : str
             The name of the file (to be) used for storing problem
             transformation on disk. NOTE: this parameter must be
             identical for all calls to \p presolve following \p
             presolve_read_specfile. It can then only be changed after
             calling presolve_terminate.
          y_sign : int
             Determines the convention of sign used for the multipliers
             associated with the general linear constraints.

             * **1** 

               All multipliers corresponding to active  inequality
               constraints are non-negative for  lower bound constraints
               and non-positive for  upper bounds constraints.

             * **-1** 

               All multipliers corresponding to active  inequality
               constraints are non-positive for  lower bound constraints
               and non-negative for  upper bounds constraints.
          inactive_y : int
             Determines whether or not the multipliers corresponding to
             constraints that are inactive at the unreduced point
             corresponding to the reduced point on input to \p
             presolve_restore_solution must be set to zero. Possible
             values are: associated with the general linear
             constraints.

             * **0** 

               All multipliers corresponding to inactive  inequality
               constraints are forced to zero,  possibly at the expense
               of deteriorating the  dual feasibility condition.

             * **1** 

               Multipliers corresponding to inactive  inequality
               constraints are left unaltered.
          z_sign : int
             Determines the convention of sign used for the dual
             variables associated with the bound constraints.

             * **1** 

               All dual variables corresponding to  active lower
               bounds are non-negative, and  non-positive for active
               upper bounds.

             * **-1** 

               All dual variables corresponding to  active lower
               bounds are non-positive, and  non-negative for active
               upper bounds.
          inactive_z : int
             Determines whether or not the dual variables corresponding
             to bounds that are inactive at the unreduced point
             corresponding to the reduced point on input to
             presolve_restore_solution must be set to zero. Possible
             values are: associated with the general linear
             constraints.

             * **0** 

               All dual variables corresponding to  inactive bounds
               are forced to zero,  possibly at the expense of
               deteriorating the  dual feasibility condition.

             * **1** 

               Dual variables corresponding to inactive  bounds are
               left unaltered.
          final_x_bounds : int
             The type of final bounds on the variables returned by the
             package. This parameter can take the values:

             * **0** 

               the final bounds are the tightest bounds  known on the
               variables (at the risk of  being redundant with other
               constraints,  which may cause degeneracy);

             * **1** 

               the best known bounds that are known to  be
               non-degenerate. This option implies  that an additional
               real workspace of size  2 * n must be allocated.

             * **2** 

               the loosest bounds that are known to  keep the problem
               equivalent to the  original problem. This option also
               implies that an additional real  workspace of size 2 * n
               must be  allocated.  NOTE: this parameter must be
               identical for all calls to presolve (except
               presolve_initialize).
          final_z_bounds : int
             The type of final bounds on the dual variables returned by
             the package. This parameter can take the values:

             * **0**

               the final bounds are the tightest bounds  known on the
               dual variables (at the risk  of being redundant with other
               constraints,  which may cause degeneracy);

             * **1**

               the best known bounds that are known to  be
               non-degenerate. This option implies  that an additional
               real workspace of size  2 * n must be allocated.

             * **2**

               the loosest bounds that are known to  keep the problem
               equivalent to the  original problem. This option also
               implies that an additional real  workspace of size 2 * n
               
             must be allocated.  NOTE: this parameter must be identical
             for all calls to presolve (except presolve_initialize).
          final_c_bounds : int
             The type of final bounds on the constraints returned by
             the package. This parameter can take the values:

             * **0**

               the final bounds are the tightest bounds  known on the
               constraints (at the risk of  being redundant with other
               constraints,  which may cause degeneracy);

             * **1**

               the best known bounds that are known to  be
               non-degenerate. This option implies  that an additional
               real workspace of size  2 * m must be allocated.

             * **2**

               the loosest bounds that are known to  keep the problem
               equivalent to the  original problem. This option also
               implies that an additional real  workspace of size 2 * n
               must be  allocated.  NOTES: 1) This parameter must be
               identical for all calls to presolve (except
               presolve_initialize). 2) If different from 0, its value
               must be identical to that of control.final_x_bounds.
          final_y_bounds : int
             The type of final bounds on the multipliers returned by
             the package. This parameter can take the values:

             * **0**
               the final bounds are the tightest bounds  known on the
               multipliers (at the risk of  being redundant with other
               constraints,  which may cause degeneracy);

             * **1**
               the best known bounds that are known to  be
               non-degenerate. This option implies  that an additional
               real workspace of size  2 * m must be allocated.

             * **2**

               the loosest bounds that are known to  keep the problem
               equivalent to the  original problem. This option also
               implies that an additional real  workspace of size 2 * n
               must be  allocated.  NOTE: this parameter must be
               identical for all calls to presolve (except
               presolve_initialize).
          check_primal_feasibility : int
             The level of feasibility check (on the values of x) at the
             start of the restoration phase. This parameter can take
             the values:

             * **0**

               no check at all;

             * **1**

               the primal constraints are recomputed at x  and a
               message issued if the computed value  does not match the
               input value, or if it is  out of bounds (if
               control.print_level >= 2);

             * **2**

               the same as for 1, but presolve is  terminated if an
               incompatibilty is detected.
          check_dual_feasibility : int
             The level of dual feasibility check (on the values of x, y
             and z) at the start of the restoration phase. This
             parameter can take the values:

             * **0**

               no check at all;

             * **1**

               the dual feasibility condition is recomputed  at ( x,
               y, z ) and a message issued if the  computed value does
               not match the input value  (if control.print_level >= 2);

             * **2**

               the same as for 1, but presolve is  terminated if an
               incompatibilty is detected.  The last two values imply the
               allocation of an additional  real workspace vector of size
               equal to the number of  variables in the reduced problem.
          pivot_tol : float
             The relative pivot tolerance above which pivoting is
             considered as numerically stable in transforming the
             coefficient matrix A. A zero value corresponds to a
             totally unsafeguarded pivoting strategy (potentially
             unstable).
          min_rel_improve : float
             The minimum relative improvement in the bounds on x, y and
             z for a tighter bound on these quantities to be accepted
             in the course of the analysis. More formally, if lower is
             the current value of the lower bound on one of the x, y or
             z, and if new_lower is a tentative tighter lower bound on
             the same quantity, it is only accepted if   new_lower >=
             lower + tol * MAX( 1, ABS( lower ) ),  where  tol =
             control.min_rel_improve.  Similarly, a tentative tighter
             upper bound new_upper only replaces the current upper
             bound upper if   new_upper <= upper - tol * MAX( 1, ABS(
             upper ) ).  Note that this parameter must exceed the
             machine precision significantly.
          max_growth_factor : float
             The maximum growth factor (in absolute value) that is
             accepted between the maximum data item in the original
             problem and any data item in the reduced problem. If a
             transformation results in this bound being exceeded, the
             transformation is skipped.

   .. function:: [optional] presolve.information()

      Provide optional output information.

      **Returns:**

      inform : dict
         dictionary containing output information:
          status : int
            The presolve exit condition. It can take the following
            values (symbol in parentheses is the related Fortran
            code):

            *  **0** (OK)  

               successful exit;

            *  **1** (MAX_NBR_TRANSF)  

               the maximum number of problem
               transformation has been reached  NOTE:  this exit is not
               really an error, since the problem  can nevertheless be
               permuted and solved. It merely  signals that further
               problem reduction could possibly  be obtained with a
               larger value of the parameter
               control.max_nbr_transforms

            * **-1** (MEMORY_FULL)  

              memory allocation failed

            * **-2** (FILE_NOT_OPENED)  

              a file intended for saving problem
              transformations  could not be opened;

            * **-3** (COULD_NOT_WRITE)  

              an IO error occurred while saving
              transformations on  the relevant disk file;

            * **-4** (TOO_FEW_BITS_PER_BYTE)  

              an integer contains less than NBRH + 1 bits.

            * **-21** (PRIMAL_INFEASIBLE)  

              the problem is primal infeasible;

            * **-22** (DUAL_INFEASIBLE)  

              the problem is dual infeasible;

            * **-23** (WRONG_G_DIMENSION)  

              the dimension of the gradient
              is incompatible with  the problem dimension;

            * **-24** (WRONG_HVAL_DIMENSION)  

              the dimension of the vector containing the entries of  
              the Hessian is erroneously specified;

            * **-25** (WRONG_HPTR_DIMENSION)  

              the dimension of the vector
              containing the addresses  of the first entry of each
              Hessian row is erroneously specified;

            * **-26** (WRONG_HCOL_DIMENSION)  

              the dimension of the vector
              containing the column  indices of the nonzero Hessian
              entries is erroneously specified;

            * **-27** (WRONG_HROW_DIMENSION) 

              the dimension of the vector
              containing the row indices  of the nonzero Hessian entries
              is erroneously specified;

            * **-28** (WRONG_AVAL_DIMENSION) 

              the dimension of the vector
              containing the entries of  the Jacobian is erroneously
              specified;

            * **-29** (WRONG_APTR_DIMENSION) 

              the dimension of the vector
              containing the addresses  of the first entry of each
              Jacobian row is erroneously specified;

            * **-30** (WRONG_ACOL_DIMENSION) 

              the dimension of the vector
              containing the column  indices of the nonzero Jacobian
              entries is erroneously specified;

            * **-31** (WRONG_AROW_DIMENSION) 

              the dimension of the vector
              containing the row indices  of the nonzero Jacobian
              entries is erroneously specified;

            * **-32** (WRONG_X_DIMENSION) 

              the dimension of the vector of
              variables is  incompatible with the problem dimension;

            * **-33** (WRONG_XL_DIMENSION) 

              the dimension of the vector of
              lower bounds on the  variables is incompatible with the
              problem dimension;

            * **-34** (WRONG_XU_DIMENSION) 

              the dimension of the vector of
              upper bounds on the  variables is incompatible with the
              problem dimension;

            * **-35** (WRONG_Z_DIMENSION) 

              the dimension of the vector of
              dual variables is  incompatible with the problem
              dimension;

            * **-36** (WRONG_ZL_DIMENSION) 

              the dimension of the vector of
              lower bounds on the dual  variables is incompatible with
              the problem dimension;

            * **-37** (WRONG_ZU_DIMENSION) 

              the dimension of the vector of
              upper bounds on the  dual variables is incompatible with
              the problem dimension;.

            * **-38** (WRONG_C_DIMENSION) 

              the dimension of the vector of
              constraints values is  incompatible with the problem
              dimension;

            * **-39** (WRONG_CL_DIMENSION) 

              the dimension of the vector of
              lower bounds on the  constraints is incompatible with the
              problem dimension;

            * **-40** (WRONG_CU_DIMENSION) 

              the dimension of the vector of
              upper bounds on the  constraints is incompatible with the
              problem dimension;

            * **-41** (WRONG_Y_DIMENSION) 

              the dimension of the vector of
              multipliers values is  incompatible with the problem
              dimension;

            * **-42** (WRONG_YL_DIMENSION) 

              the dimension of the vector of
              lower bounds on the  multipliers is incompatible with the
              problem dimension;

            * **-43** (WRONG_YU_DIMENSION) 

              the dimension of the vector of
              upper bounds on the  multipliers is incompatible with the
              problem dimension;

            * **-44** (STRUCTURE_NOT_SET) 

              the problem structure has not
              been set or has been  cleaned up before an attempt to
              analyze;

            * **-45** (PROBLEM_NOT_ANALYZED) 

              the problem has not been
              analyzed before an attempt to permute it;

            * **-46** (PROBLEM_NOT_PERMUTED) 

              the problem has not been
              permuted or fully reduced before an attempt  to restore it

            * **-47** (H_MISSPECIFIED) 

              the column indices of a row of the
              sparse Hessian are  not in increasing order, in that they
              specify an entry  above the diagonal;

            * **-48** (CORRUPTED_SAVE_FILE) 

              one of the files containing
              saved problem  transformations has been corrupted between
              writing and reading;

            * **-49** (WRONG_XS_DIMENSION) 

              the dimension of the vector of
              variables' status  is incompatible with the problem
              dimension;

            * **-50** (WRONG_CS_DIMENSION) 

              the dimension of the vector of
              constraints' status  is incompatible with the problem
              dimension;

            * **-52** (WRONG_N) 

              the problem does not contain any (active) variable;

            * **-53** (WRONG_M) 

              the problem contains a negative number of
              constraints;

            * **-54** (SORT_TOO_LONG) 

              the vectors are too long for the
              sorting routine;

            * **-55** (X_OUT_OF_BOUNDS) 

              the value of a variable that is
              obtained by  substitution from a constraint is incoherent
              with the  variable's bounds. This may be due to a
              relatively  loose accuracy on the linear constraints. Try
              to  increase control.c_accuracy.

            * **-56** (X_NOT_FEASIBLE) 

              the value of a constraint that is
              obtained by  recomputing its value on input of \p
              presolve_restore_solution  from the current x is
              incompatible with its declared value  or its bounds. This
              may caused the restored problem  to be infeasible.

            * **-57** (Z_NOT_FEASIBLE) 

              the value of a dual variable that
              is obtained by  recomputing its value on input to \p
              presolve_restore_solution  (assuming dual feasibility)
              from the current values of  $(x, y, z)$ is incompatible
              with its declared value.  This may caused the restored
              problem to be infeasible  or suboptimal.

            * **-58** (Z_CANNOT_BE_ZEROED) 

              a dual variable whose value is
              nonzero because the  corresponding primal is at an
              artificial bound cannot  be zeroed while maintaining dual
              feasibility  (on restoration). This can happen when
              $( x, y, z)$ on  input of RESTORE are not (sufficiently)
              optimal.

            * **-60** (UNRECOGNIZED_KEYWORD) 

              a keyword was not recognized
              in the analysis of the  specification file

            * **-61** (UNRECOGNIZED_VALUE) 

              a value was not recognized in
              the analysis of the specification file

            * **-63** (G_NOT_ALLOCATED) 

              the vector G has not been
              allocated although it has general values

            * **-64** (C_NOT_ALLOCATED) 

              the vector C has not been
              allocated although m > 0

            * **-65** (AVAL_NOT_ALLOCATED) 

              the vector A.val has not been
              allocated although m > 0

            * **-66** (APTR_NOT_ALLOCATED) 

              the vector A.ptr has not been
              allocated although  m > 0 and A is stored in row-wise
              sparse format

            * **-67** (ACOL_NOT_ALLOCATED) 

              the vector A.col has not been
              allocated although  m > 0 and A is stored in row-wise
              sparse format  or sparse coordinate format

            * **-68** (AROW_NOT_ALLOCATED) 

              the vector A.row has not been
              allocated although  m > 0 and A is stored in sparse
              coordinate format

            * **-69** (HVAL_NOT_ALLOCATED)  

              the vector H.val has not been
              allocated although  H.ne > 0

            * **-70** (HPTR_NOT_ALLOCATED)  

              the vector H.ptr has not been
              allocated although  H.ne > 0 and H is stored in row-wise
              sparse format

            * **-71** (HCOL_NOT_ALLOCATED)

              the vector H.col has not been
              allocated although  H.ne > 0 and H is stored in row-wise
              sparse format  or sparse coordinate format

            * **-72** (HROW_NOT_ALLOCATED)  

              the vector H.row has not been
              allocated although  H.ne > 0 and A is stored in sparse
              coordinate  format

            * **-73** (WRONG_ANE)  

              incompatible value of A_ne

            * **-74** (WRONG_HNE)

              incompatible value of H_ne.
          nbr_transforms : int
             The final number of problem transformations, as reported
             to the user at exit.
          message : str
             A few lines containing a description of the exit condition
             on exit of PRESOLVE, typically including more information
             than indicated in the description of control.status above.
             It is printed out on device errout at the end of execution
             if control.print_level >= 1.

   .. function:: presolve.finalize()

     Deallocate all internal private storage.
