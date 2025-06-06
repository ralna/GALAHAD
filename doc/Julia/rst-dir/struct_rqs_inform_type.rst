.. index:: pair: table; rqs_inform_type
.. _doxid-structrqs__inform__type:

rqs_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct rqs_inform_type{T,INT}
          status::INT
          alloc_status::INT
          factorizations::INT
          max_entries_factors::Int64
          len_history::INT
          obj::T
          obj_regularized::T
          x_norm::T
          multiplier::T
          pole::T
          dense_factorization::Bool
          hard_case::Bool
          bad_alloc::NTuple{81,Cchar}
          time::rqs_time_type{T}
          history::NTuple{100,rqs_history_type{T}}
          sls_inform::sls_inform_type{T,INT}
          ir_inform::ir_inform_type{T,INT}

.. _details-structrqs__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structrqs__inform__type_status:

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

  n and/or sigma is not positive and/or p <= 2

* **-9**

  the analysis phase of the factorization of $H + \lambda M$ failed

* **-10**

  the factorization of $H + \lambda M$ failed

* **-15**

  $M$ does not appear to be strictly diagonally dominant

* **-16**

  ill-conditioning has prevented furthr progress

.. index:: pair: variable; alloc_status
.. _doxid-structrqs__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structrqs__inform__type_factorizations:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorizations

the number of factorizations performed

.. index:: pair: variable; max_entries_factors
.. _doxid-structrqs__inform__type_max_entries_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; len_history
.. _doxid-structrqs__inform__type_len_history:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT len_history

the number of $(\|x\|_M,\lambda)$ pairs in the history

.. index:: pair: variable; obj
.. _doxid-structrqs__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structrqs__inform__type_obj_regularized:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structrqs__inform__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the $M$ -norm of $x$, $\|x\|_M$

.. index:: pair: variable; multiplier
.. _doxid-structrqs__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the Lagrange multiplier corresponding to the regularization

.. index:: pair: variable; pole
.. _doxid-structrqs__inform__type_pole:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pole

a lower bound max $(0,-\lambda_1)$, where $\lambda_1$ is the left-most eigenvalue of $(H,M)$

.. index:: pair: variable; dense_factorization
.. _doxid-structrqs__inform__type_dense_factorization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool dense_factorization

was a dense factorization used?

.. index:: pair: variable; hard_case
.. _doxid-structrqs__inform__type_hard_case:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structrqs__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structrqs__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_time_type<doxid-structrqs__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structrqs__inform__type_history:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_history_type<doxid-structrqs__history__type>` history[100]

history information

.. index:: pair: variable; sls_inform
.. _doxid-structrqs__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

cholesky information (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structrqs__inform__type_ir_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

iterative_refinement information (see ir_c documentation)

