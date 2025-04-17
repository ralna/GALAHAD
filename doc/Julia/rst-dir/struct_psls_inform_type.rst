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
.. _doxid-structpsls__inform__type_status:

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
.. _doxid-structpsls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; analyse_status
.. _doxid-structpsls__inform__type_analyse_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT analyse_status

status return from factorization

.. index:: pair: variable; factorize_status
.. _doxid-structpsls__inform__type_factorize_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorize_status

status return from factorization

.. index:: pair: variable; solve_status
.. _doxid-structpsls__inform__type_solve_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT solve_status

status return from solution phase

.. index:: pair: variable; factorization_integer
.. _doxid-structpsls__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

number of integer words to hold factors

.. index:: pair: variable; factorization_real
.. _doxid-structpsls__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

number of real words to hold factors

.. index:: pair: variable; preconditioner
.. _doxid-structpsls__inform__type_preconditioner:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

code for the actual preconditioner used (see control.preconditioner)

.. index:: pair: variable; semi_bandwidth
.. _doxid-structpsls__inform__type_semi_bandwidth:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

the actual semi-bandwidth

.. index:: pair: variable; reordered_semi_bandwidth
.. _doxid-structpsls__inform__type_reordered_semi_bandwidth:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT reordered_semi_bandwidth

the semi-bandwidth following reordering (if any)

.. index:: pair: variable; out_of_range
.. _doxid-structpsls__inform__type_out_of_range:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structpsls__inform__type_duplicates:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT duplicates

number of duplicates

.. index:: pair: variable; upper
.. _doxid-structpsls__inform__type_upper:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT upper

number of entries from the strict upper triangle

.. index:: pair: variable; missing_diagonals
.. _doxid-structpsls__inform__type_missing_diagonals:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT missing_diagonals

number of missing diagonal entries for an allegedly-definite matrix

.. index:: pair: variable; semi_bandwidth_used
.. _doxid-structpsls__inform__type_semi_bandwidth_used:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth_used

the semi-bandwidth used

.. index:: pair: variable; neg1
.. _doxid-structpsls__inform__type_neg1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT neg1

number of 1 by 1 pivots in the factorization

.. index:: pair: variable; neg2
.. _doxid-structpsls__inform__type_neg2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT neg2

number of 2 by 2 pivots in the factorization

.. index:: pair: variable; perturbed
.. _doxid-structpsls__inform__type_perturbed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perturbed

has the preconditioner been perturbed during the fctorization?

.. index:: pair: variable; fill_in_ratio
.. _doxid-structpsls__inform__type_fill_in_ratio:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T fill_in_ratio

ratio of fill in to original nonzeros

.. index:: pair: variable; norm_residual
.. _doxid-structpsls__inform__type_norm_residual:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_residual

the norm of the solution residual

.. index:: pair: variable; bad_alloc
.. _doxid-structpsls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; mc61_info
.. _doxid-structpsls__inform__type_mc61_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mc61_info[10]

the integer and real output arrays from mc61

.. index:: pair: variable; mc61_rinfo
.. _doxid-structpsls__inform__type_mc61_rinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mc61_rinfo[15]

see mc61_info

.. index:: pair: variable; time
.. _doxid-structpsls__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_time_type<doxid-structpsls__time__type>` time

times for various stages

.. index:: pair: variable; sls_inform
.. _doxid-structpsls__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform values from SLS

.. index:: pair: variable; mi28_info
.. _doxid-structpsls__inform__type_mi28_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`mi28_info<doxid-structmi28__info>` mi28_info

the output info structure from HSL's mi28

