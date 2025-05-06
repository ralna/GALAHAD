purpose
-------

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
