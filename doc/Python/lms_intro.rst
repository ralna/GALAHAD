purpose
-------

Given a sequence of vectors
$\{s_k\}$ and $\{y_k\}$ and scale factors $\{\delta_k\}$,
the ``lms`` package 
**obtains the product of a limited-memory secant approximation** 
$H_k$ (or its inverse) with a given vector,
using one of a variety of well-established formulae.

**Currently only the options and inform dictionaries are exposed**; these are 
provided and used by other GALAHAD packages with Python interfaces.
Please contact us if you would like full functionality!

See Section 4 of $GALAHAD/doc/lms.pdf for additional details.

method
------

Given a sequence of vectors $\{s_k\}$ and $\{y_k\}$ and scale factors
$\delta_k$, a limited-memory secant approximation $H_k$ is chosen so that
$H_{\max(k-m,0)} = \delta_k I$, $H_{k-j} s_{k-j} = y_{k-j}$
and $\| H_{k-j+1} - H_{k-j}\|$ is ``small'' for
$j = \min(k-1,m-1), \ldots, 0$.
Different ways of quantifying ``small'' distinguish different methods,
but the crucial observation is that it is possible to construct $H_k$
quickly from $\{s_k\}$, $\{y_k\}$ and $\delta_k$, and to apply it and
its inverse to a given vector $v$. It is also possible to apply similar
formulae to the ``shifted'' matrix $H_k + \lambda_k I$ that occurs in
trust-region methods.

reference
---------

The basic methods are those given by

  R. H. Byrd, J. Nocedal and R. B. Schnabel,
  ``Representations of quasi-Newton matrices and their use in
  limited memory methods''.
  *Mathematical Programming* **63(2)* (1994) 129--156,

with obvious extensions.
