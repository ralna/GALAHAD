.. index:: pair: struct; qpb_inform_type
.. _doxid-structqpb__inform__type:

qpb_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct qpb_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          cg_iter::INT
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          nfacts::INT
          nbacts::INT
          nmods::INT
          obj::T
          non_negligible_pivot::T
          feasible::Bool
          time::qpb_time_type{T}
          lsqp_inform::lsqp_inform_type{T,INT}
          fdc_inform::fdc_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}
          gltr_inform::gltr_inform_type{T,INT}
          fit_inform::fit_inform_type{INT}

.. _details-structqpb__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structqpb__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See QPB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structqpb__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structqpb__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structqpb__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structqpb__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structqpb__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structqpb__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structqpb__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structqpb__inform__type_nfacts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structqpb__inform__type_nbacts:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; nmods
.. _doxid-structqpb__inform__type_nmods:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nmods

the total number of factorizations which were modified to ensure that th matrix was an appropriate preconditioner

.. index:: pair: variable; obj
.. _doxid-structqpb__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by QPB_solve

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structqpb__inform__type_non_negligible_pivot:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structqpb__inform__type_feasible:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; time
.. _doxid-structqpb__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`qpb_time_type<doxid-structqpb__time__type>` time

timings (see above)

.. index:: pair: variable; lsqp_inform
.. _doxid-structqpb__inform__type_lsqp_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_inform_type<doxid-structlsqp__inform__type>` lsqp_inform

inform parameters for LSQP

.. index:: pair: variable; fdc_inform
.. _doxid-structqpb__inform__type_fdc_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structqpb__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structqpb__inform__type_gltr_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

.. index:: pair: variable; fit_inform
.. _doxid-structqpb__inform__type_fit_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fit_inform_type<doxid-structfit__inform__type>` fit_inform

return information from FIT

