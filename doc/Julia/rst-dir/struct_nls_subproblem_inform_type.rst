.. index:: pair: struct; nls_subproblem_inform_type
.. _doxid-structnls__subproblem__inform__type:

nls_subproblem_inform_type structure
------------------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct nls_subproblem_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          bad_eval::NTuple{13,Cchar}
          iter::INT
          cg_iter::INT
          c_eval::INT
          j_eval::INT
          h_eval::INT
          factorization_max::INT
          factorization_status::INT
          max_entries_factors::Int64
          factorization_integer::Int64
          factorization_real::Int64
          factorization_average::T
          obj::T
          norm_c::T
          norm_g::T
          weight::T
          time::nls_time_type{T}
          rqs_inform::rqs_inform_type{T,INT}
          glrt_inform::glrt_inform_type{T,INT}
          psls_inform::psls_inform_type{T,INT}
          bsc_inform::bsc_inform_type{T,INT}
          roots_inform::roots_inform_type{INT}

.. _details-structnls__subproblem__inform__type:

detailed documentation
----------------------

subproblem_inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structnls__subproblem__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See NLS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structnls__subproblem__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structnls__subproblem__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structnls__subproblem__inform__type_bad_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structnls__subproblem__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structnls__subproblem__inform__type_cg_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of CG iterations performed

.. index:: pair: variable; c_eval
.. _doxid-structnls__subproblem__inform__type_c_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT c_eval

the total number of evaluations of the residual function c(x)

.. index:: pair: variable; j_eval
.. _doxid-structnls__subproblem__inform__type_j_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT j_eval

the total number of evaluations of the Jacobian J(x) of c(x)

.. index:: pair: variable; h_eval
.. _doxid-structnls__subproblem__inform__type_h_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the scaled Hessian H(x,y) of c(x)

.. index:: pair: variable; factorization_max
.. _doxid-structnls__subproblem__inform__type_factorization_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; factorization_status
.. _doxid-structnls__subproblem__inform__type_factorization_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; max_entries_factors
.. _doxid-structnls__subproblem__inform__type_max_entries_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structnls__subproblem__inform__type_factorization_integer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structnls__subproblem__inform__type_factorization_real:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; factorization_average
.. _doxid-structnls__subproblem__inform__type_factorization_average:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorization_average

the average number of factorizations per sub-problem solve

.. index:: pair: variable; obj
.. _doxid-structnls__subproblem__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function $\frac{1}{2}\|c(x)\|^2_W$ at the best estimate the solution, x, determined by NLS_solve

.. index:: pair: variable; norm_c
.. _doxid-structnls__subproblem__inform__type_norm_c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_c

the norm of the residual $\|c(x)\|_W$ at the best estimate of the solution x, determined by NLS_solve

.. index:: pair: variable; norm_g
.. _doxid-structnls__subproblem__inform__type_norm_g:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_g

the norm of the gradient of $\|c(x)\|_W$ of the objective function at the best estimate, x, of the solution determined by NLS_solve

.. index:: pair: variable; weight
.. _doxid-structnls__subproblem__inform__type_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight

the final regularization weight used

.. index:: pair: variable; time
.. _doxid-structnls__subproblem__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`nls_time_type<doxid-structnls__time__type>` time

timings (see above)

.. index:: pair: variable; rqs_inform
.. _doxid-structnls__subproblem__inform__type_rqs_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

inform parameters for RQS

.. index:: pair: variable; glrt_inform
.. _doxid-structnls__subproblem__inform__type_glrt_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` glrt_inform

inform parameters for GLRT

.. index:: pair: variable; psls_inform
.. _doxid-structnls__subproblem__inform__type_psls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; bsc_inform
.. _doxid-structnls__subproblem__inform__type_bsc_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` bsc_inform

inform parameters for BSC

.. index:: pair: variable; roots_inform
.. _doxid-structnls__subproblem__inform__type_roots_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`roots_inform_type<doxid-structroots__inform__type>` roots_inform

inform parameters for ROOTS

