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
.. _doxid-structllsr__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

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
.. _doxid-structllsr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structllsr__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorizations

the number of factorizations performed

.. index:: pair: variable; len_history
.. _doxid-structllsr__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT len_history

the number of ($\|x\|_S$, $\lambda$) pairs in the history

.. index:: pair: variable; r_norm
.. _doxid-structllsr__inform__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

corresponding value of the two-norm of the residual, $\|A x(\lambda) - b\|$

.. index:: pair: variable; x_norm
.. _doxid-structllsr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the S-norm of x, $\|x\|_S$

.. index:: pair: variable; multiplier
.. _doxid-structllsr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the multiplier corresponding to the regularization term

.. index:: pair: variable; bad_alloc
.. _doxid-structllsr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structllsr__inform__type_1ace18e9a0877156e432cc23c7d5799dd6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`llsr_time_type<doxid-structllsr__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structllsr__inform__type_1a13047d24b0cf3469a41cc14c364d3587:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`llsr_history_type<doxid-structllsr__history__type>` history[100]

history information

.. index:: pair: variable; sbls_inform
.. _doxid-structllsr__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_inform
.. _doxid-structllsr__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structllsr__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from the iterative refinement for definite system solves (see ir_c documentation)

