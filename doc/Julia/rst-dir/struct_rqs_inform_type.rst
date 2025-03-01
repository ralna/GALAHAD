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
.. _doxid-structrqs__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

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
.. _doxid-structrqs__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structrqs__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorizations

the number of factorizations performed

.. index:: pair: variable; max_entries_factors
.. _doxid-structrqs__inform__type_1a177e429e737cfa2cd3df051a65fcfb68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; len_history
.. _doxid-structrqs__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT len_history

the number of $(\|x\|_M,\lambda)$ pairs in the history

.. index:: pair: variable; obj
.. _doxid-structrqs__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structrqs__inform__type_1a1631e243108715d623e2ddb83310fa33:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structrqs__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the $M$ -norm of $x$, $\|x\|_M$

.. index:: pair: variable; multiplier
.. _doxid-structrqs__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the Lagrange multiplier corresponding to the regularization

.. index:: pair: variable; pole
.. _doxid-structrqs__inform__type_1ad2dc9016b1d2b00a970ec28129f7000d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pole

a lower bound max $(0,-\lambda_1)$, where $\lambda_1$ is the left-most eigenvalue of $(H,M)$

.. index:: pair: variable; dense_factorization
.. _doxid-structrqs__inform__type_1a107cef1ccaad53efc9d7a578d400f324:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool dense_factorization

was a dense factorization used?

.. index:: pair: variable; hard_case
.. _doxid-structrqs__inform__type_1a22215075b7081ccac9f121daf07a0f7e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structrqs__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structrqs__inform__type_1a7ea9c3c8923491cb7b82bc1e2eaa865e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_time_type<doxid-structrqs__time__type>` time

time information

.. index:: pair: variable; history
.. _doxid-structrqs__inform__type_1a8a8041f7210288f13c6fa994dfc73aab:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_history_type<doxid-structrqs__history__type>` history[100]

history information

.. index:: pair: variable; sls_inform
.. _doxid-structrqs__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

cholesky information (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structrqs__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

iterative_refinement information (see ir_c documentation)

