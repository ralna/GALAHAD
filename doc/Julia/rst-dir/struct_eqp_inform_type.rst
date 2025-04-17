.. index:: pair: struct; eqp_inform_type
.. _doxid-structeqp__inform__type:

eqp_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct eqp_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          cg_iter::INT
          cg_iter_inter::INT
          factorization_integer::Int64
          factorization_real::Int64
          obj::T
          time::eqp_time_type{T}
          fdc_inform::fdc_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}
          gltr_inform::gltr_inform_type{T,INT}

.. _details-structeqp__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structeqp__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See EQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structeqp__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structeqp__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; cg_iter
.. _doxid-structeqp__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; cg_iter_inter
.. _doxid-structeqp__inform__type_cg_iter_inter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter_inter

see cg_iter

.. index:: pair: variable; factorization_integer
.. _doxid-structeqp__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structeqp__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; obj
.. _doxid-structeqp__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by QPB_solve

.. index:: pair: variable; time
.. _doxid-structeqp__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`eqp_time_type<doxid-structeqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structeqp__inform__type_fdc_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structeqp__inform__type_sbls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structeqp__inform__type_gltr_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

return information from GLTR

