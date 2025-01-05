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
.. _doxid-structsbls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See SBLS_form_and_factorize for details

.. index:: pair: variable; alloc_status
.. _doxid-structsbls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structsbls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; sort_status
.. _doxid-structsbls__inform__type_1acb9e43ddb17591d930fc31faa3e6f69f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sort_status

the return status from the sorting routines

.. index:: pair: variable; factorization_integer
.. _doxid-structsbls__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structsbls__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; preconditioner
.. _doxid-structsbls__inform__type_1adf7719f1a4491459e361e80a00c55656:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

the preconditioner used

.. index:: pair: variable; factorization
.. _doxid-structsbls__inform__type_1a108359f1209601e6c6074c215e3abd8b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization

the factorization used

.. index:: pair: variable; d_plus
.. _doxid-structsbls__inform__type_1a9e94a7e5692a82e8c239c857be5b80ea:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT d_plus

how many of the diagonals in the factorization are positive

.. index:: pair: variable; rank
.. _doxid-structsbls__inform__type_1a6cfd95afd0afebd625b889fb6e58371c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

the computed rank of $A$

.. index:: pair: variable; rank_def
.. _doxid-structsbls__inform__type_1aad74061a53e6daf7ca65b2e82290871b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool rank_def

is the matrix A rank defficient?

.. index:: pair: variable; perturbed
.. _doxid-structsbls__inform__type_1a6e04ee4d6dc38d2c5231d39d4f21be75:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perturbed

has the used preconditioner been perturbed to guarantee correct inertia?

.. index:: pair: variable; iter_pcg
.. _doxid-structsbls__inform__type_1a1694e6d072d1a8d796f85f4da8a054af:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter_pcg

the total number of projected CG iterations required

.. index:: pair: variable; norm_residual
.. _doxid-structsbls__inform__type_1a1f77ff3a30a89cc31d4de01c54343e86:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_residual

the norm of the residual

.. index:: pair: variable; alternative
.. _doxid-structsbls__inform__type_1a48c07c7da1803ed8af25ca949f4854b5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool alternative

has an "alternative" $y$ : $K y = 0$ and $y^T c > 0$ been found when trying to solve $K y = c$ for generic $K$?

.. index:: pair: variable; time
.. _doxid-structsbls__inform__type_1aaa565e03cbc8470593f946cf00beb639:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_time_type<doxid-structsbls__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structsbls__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

inform parameters from the GALAHAD package SLS used

.. index:: pair: variable; uls_inform
.. _doxid-structsbls__inform__type_1aa39eb0d7b50d4a858849f8ef652ae84c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

inform parameters from the GALAHAD package ULS used

