purpose
-------

The ``sha`` package
**finds a component-wise secant approximation to the Hessian matrix** $H(x)$,
for which $(H(x))_{i,j} = \partial f^2 (x) / \partial x_i \partial x_j$,
$1 \leq i, j \leq n$,
using values of the gradient $g(x) = \nabla_x f(x)$
of the function $f(x)$ of $n$ unknowns $x = (x_1, \ldots, x_n)^T$
at a sequence of given distinct $\{x^{(k)}\}$, $k \geq 0$.
More specifically, given **differences**
$$s^{(k)} = x^{(k+1)} - x^{(k)} \;\;\mbox{and}\;\;
y^{(k)} = g(x^{(k+1)}) - g(x^{(k)})$$

the package aims to find an approximation $B$ to $H(x)$ for
which the secant conditions $B s^{(k)} \approx y^{(k)}$ hold for
a chosen set of values $k$.
The methods provided take advantage of the entries in the Hessian that
are known to be zero.

The package is particularly intended to allow gradient-based
optimization methods, that generate iterates
$x^{(k+1)} = x^{(k)} + s^{(k)}$ based upon the values $g( x^{(k)})$
for $k \geq 0$, to build a suitable approximation to the Hessian
$H(x^{(k+1)})$. This then gives the method an opportunity to
accelerate the iteration using the Hessian approximation.

See Section 4 of $GALAHAD/doc/sha.pdf for additional details.

method
------

The package computes the entries in the each row of $B$ one at a time.
The entries $b_{ij}$ in row $i$ may be chosen to

$$(1) \;\;\; \min_{b_{i,j}}\!\mbox{imize} \;\; \sum_{k \in {\cal I}_i}
\left[ \sum_{{\scriptscriptstyle \mbox{nonzeros}}\; j}
b_{i,j} s_j^{(k)} - y_i^{(k)} \right]^2,$$

where ${\cal I}_i$ is ideally chosen to be sufficiently large so that (1)
has a unique minimizer. Since this requires that there are at least
as many $(s^{(k)}, y^{(k)})$ pairs as the maximum number of nonzeros
in any row, this may be prohibitive in some cases. We might then be content
with a minimum-norm (under-determined) least-squares solution; each row
may then be processed in parallel. Or, we may
take advantage of the symmetry of the Hessian, and note that if we
have already found the values in row $j$, then the value $b_{i,j} = b_{j,i}$
in (1) is known before we process row $i$. Thus by ordering the rows
and exploiting symmetry we may reduce the numbers of unknowns in
future unprocessed rows.

In the analysis phase, we order the rows by constructing the connectivity
graph---a graph comprising nodes $1$ to $n$ and edges connecting
nodes $i$ and $j$ if $h_{i,j}$ is everywhere nonzero---of $H(x)$.
The nodes are ordered by increasing degree (that is, the number of edges
emanating from the node) using a bucket sort. The row chosen to be
ordered next corresponds to a node of minimum degree, the node
is removed from the graph, the degrees updated efficiently, and the
process repeated until all rows have been ordered. This often leads
to a significant reduction in the numbers of unknown values in each
row as it is processed in turn, but numerical rounding can lead to
inaccurate values in some cases. A useful remedy is to process all
rows for which there are sufficient $(s^{(k)}, y^{(k)})$ as before,
and then process the remaining rows taking into account the symmetry.
That is, the rows and columns are rearranged so that the matrix
is in block form
$$B = \begin{pmatrix} B_{11} & B_{12} \\ B^T_{12} & B_{22}\end{pmatrix},$$

the $( B_{11} \;\; B_{12})$ rows are processed without regard
for symmetry but give the $2,1$ block $B^T_{12}$,
and finally
the $2,2$ block $B_{22}$ is processed knowing $B^T_{12}$
again without respecting symmetry. The rows in blocks
$( B_{11} \;\; B_{12})$ and $B_{22}$ may be computed in parallel.
It is also possible to generalise this so that $B$ is decomposed into
$r$ blocks, and the blocks processed one at a time recursively using the
symmetry from previos rows.
More details of the precise algorithms (Algorithms 2.1--2.5)
are given in the reference below. The linear least-squares problems (1)
themselves are solved by a choice of LAPACK packages.

references
----------

The method is described in detail in

  J. M. Fowkes, N. I. M. Gould and J. A. Scott,
  "Approximating large-scale Hessians using secant equations".
  Preprint TR-2024-001, Rutherford Appleton Laboratory.
