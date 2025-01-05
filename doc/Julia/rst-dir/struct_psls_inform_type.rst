.. index:: pair: table; psls_inform_type
.. _doxid-structpsls__inform__type:

psls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct psls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          analyse_status::INT
          factorize_status::INT
          solve_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          preconditioner::INT
          semi_bandwidth::INT
          reordered_semi_bandwidth::INT
          out_of_range::INT
          duplicates::INT
          upper::INT
          missing_diagonals::INT
          semi_bandwidth_used::INT
          neg1::INT
          neg2::INT
          perturbed::Bool
          fill_in_ratio::T
          norm_residual::T
          bad_alloc::NTuple{81,Cchar}
          mc61_info::NTuple{10,INT}
          mc61_rinfo::NTuple{15,T}
          time::psls_time_type{T}
          sls_inform::sls_inform_type{T,INT}
          mi28_info::mi28_info{T,INT}

.. _details-structpsls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structpsls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status:

* **0**

  success

* **-1**

  allocation error

* **-2**

  deallocation error

* **-3**

  matrix data faulty (.n < 1, .ne < 0)

.. index:: pair: variable; alloc_status
.. _doxid-structpsls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; analyse_status
.. _doxid-structpsls__inform__type_1ae38019a70cc3dffa90bd881451c6cf1b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT analyse_status

status return from factorization

.. index:: pair: variable; factorize_status
.. _doxid-structpsls__inform__type_1a08c6a015b4a7616dffae6ab4972af1ab:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorize_status

status return from factorization

.. index:: pair: variable; solve_status
.. _doxid-structpsls__inform__type_1aaab916515d75c0f3abbc4a250381708b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT solve_status

status return from solution phase

.. index:: pair: variable; factorization_integer
.. _doxid-structpsls__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

number of integer words to hold factors

.. index:: pair: variable; factorization_real
.. _doxid-structpsls__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

number of real words to hold factors

.. index:: pair: variable; preconditioner
.. _doxid-structpsls__inform__type_1adf7719f1a4491459e361e80a00c55656:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

code for the actual preconditioner used (see control.preconditioner)

.. index:: pair: variable; semi_bandwidth
.. _doxid-structpsls__inform__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

the actual semi-bandwidth

.. index:: pair: variable; reordered_semi_bandwidth
.. _doxid-structpsls__inform__type_1a626e2d4fb989dd770239efa3be051e0a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT reordered_semi_bandwidth

the semi-bandwidth following reordering (if any)

.. index:: pair: variable; out_of_range
.. _doxid-structpsls__inform__type_1a8daa2a776cae6116e9f14e2b009430a5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structpsls__inform__type_1a4266bf48aafe2914b08e60d6ef9cf446:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT duplicates

number of duplicates

.. index:: pair: variable; upper
.. _doxid-structpsls__inform__type_1a0a1a19aadb8cf4f2b05d37a8798b667c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT upper

number of entries from the strict upper triangle

.. index:: pair: variable; missing_diagonals
.. _doxid-structpsls__inform__type_1a8d33160feb6e388439a1d38641b00b3d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT missing_diagonals

number of missing diagonal entries for an allegedly-definite matrix

.. index:: pair: variable; semi_bandwidth_used
.. _doxid-structpsls__inform__type_1a981530ec3c99dba9d28c74cdacca6bbf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth_used

the semi-bandwidth used

.. index:: pair: variable; neg1
.. _doxid-structpsls__inform__type_1ac7c6e49ad4048d11de36fcc4ce540aba:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT neg1

number of 1 by 1 pivots in the factorization

.. index:: pair: variable; neg2
.. _doxid-structpsls__inform__type_1a8fe93b02eb981bd3300337dee7835d86:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT neg2

number of 2 by 2 pivots in the factorization

.. index:: pair: variable; perturbed
.. _doxid-structpsls__inform__type_1a6e04ee4d6dc38d2c5231d39d4f21be75:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perturbed

has the preconditioner been perturbed during the fctorization?

.. index:: pair: variable; fill_in_ratio
.. _doxid-structpsls__inform__type_1a255e78721c3559caab816b9e6e72a6d4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T fill_in_ratio

ratio of fill in to original nonzeros

.. index:: pair: variable; norm_residual
.. _doxid-structpsls__inform__type_1a1f77ff3a30a89cc31d4de01c54343e86:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_residual

the norm of the solution residual

.. index:: pair: variable; bad_alloc
.. _doxid-structpsls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; mc61_info
.. _doxid-structpsls__inform__type_1ad29411cd0e18c59e43b474314a2adbe8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mc61_info[10]

the integer and real output arrays from mc61

.. index:: pair: variable; mc61_rinfo
.. _doxid-structpsls__inform__type_1a6966776cf11a3b9c447f7a1c9621152f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mc61_rinfo[15]

see mc61_info

.. index:: pair: variable; time
.. _doxid-structpsls__inform__type_1a4e85e8fc22799defca71ba5c448216ed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_time_type<doxid-structpsls__time__type>` time

times for various stages

.. index:: pair: variable; sls_inform
.. _doxid-structpsls__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform values from SLS

.. index:: pair: variable; mi28_info
.. _doxid-structpsls__inform__type_1aa5913427f989eb08152b78bf6390c0b9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`mi28_info<doxid-structmi28__info>` mi28_info

the output info structure from HSL's mi28

