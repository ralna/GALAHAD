.. index:: pair: table; sbls_inform_type
.. _doxid-structsbls__inform__type:

sbls_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sbls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          sort_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          preconditioner::INT
          factorization::INT
          d_plus::INT
          rank::INT
          rank_def::Bool
          perturbed::Bool
          iter_pcg::INT
          norm_residual::T
          alternative::Bool
          time::sbls_time_type{T}
          sls_inform::sls_inform_type{T,INT}
          uls_inform::uls_inform_type{T,INT}

.. _details-structsbls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsbls__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See SBLS_form_and_factorize for details

.. index:: pair: variable; alloc_status
.. _doxid-structsbls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structsbls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; sort_status
.. _doxid-structsbls__inform__type_sort_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sort_status

the return status from the sorting routines

.. index:: pair: variable; factorization_integer
.. _doxid-structsbls__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structsbls__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; preconditioner
.. _doxid-structsbls__inform__type_preconditioner:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

the preconditioner used

.. index:: pair: variable; factorization
.. _doxid-structsbls__inform__type_factorization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization

the factorization used

.. index:: pair: variable; d_plus
.. _doxid-structsbls__inform__type_d_plus:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT d_plus

how many of the diagonals in the factorization are positive

.. index:: pair: variable; rank
.. _doxid-structsbls__inform__type_rank:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

the computed rank of $A$

.. index:: pair: variable; rank_def
.. _doxid-structsbls__inform__type_rank_def:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool rank_def

is the matrix A rank defficient?

.. index:: pair: variable; perturbed
.. _doxid-structsbls__inform__type_perturbed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perturbed

has the used preconditioner been perturbed to guarantee correct inertia?

.. index:: pair: variable; iter_pcg
.. _doxid-structsbls__inform__type_iter_pcg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter_pcg

the total number of projected CG iterations required

.. index:: pair: variable; norm_residual
.. _doxid-structsbls__inform__type_norm_residual:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_residual

the norm of the residual

.. index:: pair: variable; alternative
.. _doxid-structsbls__inform__type_alternative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool alternative

has an "alternative" $y$ : $K y = 0$ and $y^T c > 0$ been found when trying to solve $K y = c$ for generic $K$?

.. index:: pair: variable; time
.. _doxid-structsbls__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_time_type<doxid-structsbls__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structsbls__inform__type_sls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters from the GALAHAD package SLS used

.. index:: pair: variable; uls_inform
.. _doxid-structsbls__inform__type_uls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

inform parameters from the GALAHAD package ULS used

