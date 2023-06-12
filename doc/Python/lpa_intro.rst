purpose
-------

The ``lpa`` package uses the **simplex method** to solve a
given **linear program** (LP).
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

See Section 4 of $GALAHAD/doc/lpa.pdf for a brief description of the
method employed and other details.

**N.B.** The package is simply a sophisticated interface to the HSL package
LA04, and requires that a user has obtained the latter. LA04 is not
included in GALAHAD but is available without charge to recognised
academics, see http://www.hsl.rl.ac.uk/catalogue/la04.html. If LA04 is
unavailable, the interior- point linear programming package ``lpb``
is an alternative.

terminology
-----------

Any required solution $x$ necessarily satisfies
the **primal optimality conditions**
$$A x = c$$
and
$$c_l \leq c \leq c_u, \;\; x_l \leq x \leq x_u,$$
the **dual optimality conditions**
$$g = A^{T} y + z,\;\;  y = y_l + y_u \;\;\mbox{and}\;\; z = z_l + z_u,$$
and
$$y_l \geq 0, \;\; y_u \leq 0, \;\; z_l \geq 0 \;\;\mbox{and}\;\; z_u \leq 0,$$
and the **complementary slackness conditions**
$$( A x - c_l )^{T} y_l = 0,\;\; ( A x - c_u )^{T} y_u = 0,\;\;
(x -x_l )^{T} z_l = 0 \;\;\mbox{and}\;\;(x -x_u )^{T} z_u = 0,$$
where the vectors $y$ and $z$ are known as the **Lagrange multipliers** for
the general linear constraints, and the **dual variables** for the bounds,
respectively, and where the vector inequalities hold component-wise.

The so-called **dual** to this problem is another linear program
$$- \mbox{minimize} \;\; c^{lT} y^l + c^{uT} y^u +
x^{lT} z^l + x^{uT} z^u + f
\;\; \mbox{subject to the dual optimality conditions}$$
that uses the same data. The solution to the two problems, it is exists,
is the same, but if one is infeasible, the other is unbounded. It can be
more efficient to solve the dual, particularly if $m$ is much larger than $n$.

method
------

The bulk of the work is peformed by the HSL package *LA04*. The
main subbroutine from this package requires that the input problem
be transformed into the ``standard form''
$$\begin{array}{rl} 
\mbox{minimize} & g'^T x' \\
\mbox{subject to} & A' x' = b \\
& l_i \leq x'_i \leq u_i, \;\;  (i\leq k) \\
\mbox{and}  & x'_l \geq 0, \;\; (i \geq l) \end{array}$$
by introducing slack an surpulus variables, reordering and
removing fixed variables and free constraints. The resulting
problem involves $n'$ unknowns and $m'$ general constraints.
In order to deal with the possibility that the general constraints
are inconsistent or not of full rank,
*LA04* introduces additional ``artifical'' variables $v$,
replaces the constraints of the standard-form LP by the enlaarged set
$$A' x' + v = b,$$
and gradually encourages $v$ to zero as a first solution phase.

Once a selection of $m'$ independent (*non-basic*) variables
is made, the enlarged constraints determine the remaining $m'$
dependent ({*basic*) variables. The *simplex method*
is a scheme for systematically adjusting the choice of basic
and non-basic variables until a set which defines an optimal
solution of the standard-form LP is obtained. Each
iteration of the simplex method requires the solution of a number
of sets of linear equations whose coefficient matrix is the
*basis* matrix $B$, made up of the columns of $[A' \;\; I]$
corresponding to the
basic variables, or its transpose $B^T$. As the basis matrices for
consecutive iterations are closely related, it is normally
advantageous to update (rather than recompute) their factorizations
as the computation proceeds.
If an initial basis is not provided by the user,
a set of basic variables which provide a (permuted)
triangular basis matrix is found by the simple crash algorithm of
Gould and Reid (1989), and initial steepest-edge weights are calculated.

Phases one (finding a feasible solution)
and two (solving the standard-form LP) of the simplex method are applied, 
as appropriate, with the choice of entering variable as described
by Goldfarb and Reid (1977) and the choice of leaving variable as proposed 
by Harris (1973). Refactorizations of the basis matrix are performed 
whenever doing so will reduce the average iteration time or there 
is insufficient memory for its factors.
The reduced cost for the entering variable is computed afresh. If it
is found to be of a different sign from the recurred value or more than
10% different in magnitude, a fresh computation of all the reduced
costs is performed. Details of the factorization and updating procedures 
are given by Reid (1982). Iterative refinement is encouraged for the 
basic solution and for the reduced costs after each factorization
of the basis matrix and when they are recomputed at the end of phase 1.

references
----------

  D. Goldfarb and J. K. Reid,
  ``A practicable steepest-edge simplex algorithm''.
  *Mathematical Programming* **12** (1977) 361-371.

  N. I. M. Gould and J. K. Reid,
  ``New crash procedures for large systems of linear constraints''.
  *Mathematical Programming* **45** (1989) 475-501.

  P. M. J. Harris,
  ``Pivot selection methods of the Devex LP code''.
  *Mathematical Programming* **5** (1973) 1-28.

  J. K. Reid,
  ``A sparsity-exploiting variant of the Bartels-Golub
  decomposition for linear-programming bases''.
  *Mathematical Programming* **24** (1982) 55-69.
