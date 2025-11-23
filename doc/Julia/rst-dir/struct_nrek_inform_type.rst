.. index:: pair: table; nrek_inform_type
.. _doxid-structnrek__inform__type:

nrek_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct nrek_inform_type{T,INT}
          status::INT
          alloc_status::INT
          iter::INT
          n_vec::INT
          obj::T
          x_norm::T
          multiplier::T
          weight::T
          next_weight::T
          error::T
          bad_alloc::NTuple{81,Cchar}
          time::nrek_time_type{T}
          sls_inform::sls_inform_type{T,INT}
          sls_s_inform::sls_inform_type{T,INT}
          rqs_inform::rqs_inform_type{T,INT}

.. _details-structnrek__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnrek__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status:

* **0**

  the solution has been found

* **-1**

  an array allocation has failed

* **-2**

  an array deallocation has failed

* **-3**

  n and/or $\Delta$ is not positive, or the requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal', 'scaled_identity',  'identity', 'zero' or 'none' has been violated.

* **-9**

  the analysis phase of the factorization failed; the return status from the factorization package is given by inform.sls_inform.status or inform.sls_s_inform.status as appropriate

* **-10**

  the factorization failed; the return status from the factorization package is given by inform.sls_inform.status or inform.sls_s_inform.status as appropriate

* **-11**

  the solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given by inform.sls_inform.status or inform.sls_s_inform.status as appropriate

* **-15**

  $S$ does not appear to be strictly diagonally dominant

* **-16**

  ill-conditioning has prevented further progress

* **-18**

  too many iterations have been required. This may happen if control.eks max is too small, but may also be symptomatic of  a badly scaled problem.

* **-31** 

  a resolve call has been made before an initial call (see control.new_weight and control.new_values)

* **-38** 

  an error occurred in a call to an LAPACK subroutine

.. index:: pair: variable; alloc_status
.. _doxid-structnrek__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; iter
.. _doxid-structnrek__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; n_vec
.. _doxid-structnrek__inform__type_n_vec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT n_vec

the number of orthogonal vectors required (the dimension of the extended-Krylov subspace)

.. index:: pair: variable; obj
.. _doxid-structnrek__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the quadratic function $f + c^T x + \frac{1}{2} x^T H x$

.. index:: pair: variable; obj_regularized
.. _doxid-structnrek__inform__type_ob_regularizedj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_regularized

the value of the regularized quadratic function $r(x)$

.. index:: pair: variable; x_norm
.. _doxid-structnrek__inform__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the $S$ -norm of $x$, $||x||_S$

.. index:: pair: variable; multiplier
.. _doxid-structnrek__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the Lagrange multiplier associated with the regularization

.. index:: pair: variable; weight
.. _doxid-structnrek__inform__type_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight

the value of the current weight

.. index:: pair: variable; next_weight
.. _doxid-structnrek__inform__type_next_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T next_weight

the value of the the proposed next weight to be used if the current weight proves to be too small (see inform.increase).

.. index:: pair: variable; error
.. _doxid-structnrek__inform__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T error

the value of the norm of the maximum relative residual error, $\|(H+\lambda S) x + c\|/\max(1,\|c\|)$

.. index:: pair: variable; bad_alloc
.. _doxid-structnrek__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structnrek__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`nrek_time_type<doxid-structnrek__time__type>` time

time information

.. index:: pair: variable; sls_inform
.. _doxid-structnrek__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

Cholesky information for factorization and solves with $H$ (see sls_c documentation)

.. index:: pair: variable; sls_s_inform
.. _doxid-structnrek__inform__type_sls_s_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_s_inform

Cholesky information for factorization and solves with $S$ (see sls_c documentation)

.. index:: pair: variable; rqs_inform
.. _doxid-structnrek__inform__type_rqs_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

diagonal norm-regularization solve information (see rqs_c documentation)

