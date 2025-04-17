.. index:: pair: struct; trb_inform_type
.. _doxid-structtrb__inform__type:

trb_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct trb_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          cg_iter::INT
          cg_maxit::INT
          f_eval::INT
          g_eval::INT
          h_eval::INT
          n_free::INT
          factorization_max::INT
          factorization_status::INT
          max_entries_factors::Int64
          factorization_integer::Int64
          factorization_real::Int64
          obj::T
          norm_pg::T
          radius::T
          time::trb_time_type{T}
          trs_inform::trs_inform_type{T,INT}
          gltr_inform::gltr_inform_type{T,INT}
          psls_inform::psls_inform_type{T,INT}
          lms_inform::lms_inform_type{T,INT}
          lms_inform_prec::lms_inform_type{T,INT}
          sha_inform::sha_inform_type{T,INT}

.. _details-structtrb__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structtrb__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See TRB_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structtrb__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structtrb__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structtrb__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structtrb__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of CG iterations performed

.. index:: pair: variable; cg_maxit
.. _doxid-structtrb__inform__type_cg_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_maxit

the maximum number of CG iterations allowed per iteration

.. index:: pair: variable; f_eval
.. _doxid-structtrb__inform__type_f_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structtrb__inform__type_g_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structtrb__inform__type_h_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; n_free
.. _doxid-structtrb__inform__type_n_free:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT n_free

the number of variables that are free from their bounds

.. index:: pair: variable; factorization_max
.. _doxid-structtrb__inform__type_factorization_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structtrb__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structtrb__inform__type_max_entries_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structtrb__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structtrb__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; obj
.. _doxid-structtrb__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by TRB_solve

.. index:: pair: variable; norm_pg
.. _doxid-structtrb__inform__type_norm_pg:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by TRB_solve

.. index:: pair: variable; radius
.. _doxid-structtrb__inform__type_radius:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T radius

the current value of the trust-region radius

.. index:: pair: variable; time
.. _doxid-structtrb__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trb_time_type<doxid-structtrb__time__type>` time

timings (see above)

.. index:: pair: variable; trs_inform
.. _doxid-structtrb__inform__type_trs_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trs_inform_type<doxid-structtrs__inform__type>` trs_inform

inform parameters for TRS

.. index:: pair: variable; gltr_inform
.. _doxid-structtrb__inform__type_gltr_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gltr_inform_type<doxid-structgltr__inform__type>` gltr_inform

inform parameters for GLTR

.. index:: pair: variable; psls_inform
.. _doxid-structtrb__inform__type_psls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; lms_inform
.. _doxid-structtrb__inform__type_lms_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform

inform parameters for LMS

.. index:: pair: variable; lms_inform_prec
.. _doxid-structtrb__inform__type_lms_inform_prec:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform_prec

inform parameters for LMS used for preconditioning

.. index:: pair: variable; sha_inform
.. _doxid-structtrb__inform__type_sha_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sha_inform_type<doxid-structsha__inform__type>` sha_inform

inform parameters for SHA

