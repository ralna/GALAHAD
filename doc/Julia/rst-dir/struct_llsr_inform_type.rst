.. index:: pair: table; llsr_inform_type
.. _doxid-structllsr__inform__type:

llsr_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct llsr_inform_type{T,INT}
          status::INT
          alloc_status::INT
          factorizations::INT
          len_history::INT
          r_norm::T
          x_norm::T
          multiplier::T
          bad_alloc::NTuple{81,Cchar}
          time::llsr_time_type{T}
          history::NTuple{100,llsr_history_type{T}}
          sbls_inform::sbls_inform_type{T,INT}
          sls_inform::sls_inform_type{T,INT}
          ir_inform::ir_inform_type{T,INT}

.. _details-structllsr__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structllsr__inform__type_status:

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

  n and/or Delta is not positive

* **-10**

  the factorization of $K(\lambda)$ failed

* **-15**

  $S$ does not appear to be strictly diagonally dominant

* **-16**

  ill-conditioning has prevented furthr progress

.. index:: pair: variable; alloc_status
.. _doxid-structllsr__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structllsr__inform__type_factorizations:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorizations

the number of factorizations performed

.. index:: pair: variable; len_history
.. _doxid-structllsr__inform__type_len_history:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT len_history

the number of ($\|x\|_S$, $\lambda$) pairs in the history

.. index:: pair: variable; r_norm
.. _doxid-structllsr__inform__type_r_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

corresponding value of the two-norm of the residual, $\|A x(\lambda) - b\|$

.. index:: pair: variable; x_norm
.. _doxid-structllsr__inform__type_x_norm:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the S-norm of x, $\|x\|_S$

.. index:: pair: variable; multiplier
.. _doxid-structllsr__inform__type_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the multiplier corresponding to the regularization term

.. index:: pair: variable; bad_alloc
.. _doxid-structllsr__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structllsr__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`llsr_time_type<doxid-structllsr__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structllsr__inform__type_history:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`llsr_history_type<doxid-structllsr__history__type>` history[100]

history information

.. index:: pair: variable; sbls_inform
.. _doxid-structllsr__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_inform
.. _doxid-structllsr__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structllsr__inform__type_ir_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from the iterative refinement for definite system solves (see ir_c documentation)

