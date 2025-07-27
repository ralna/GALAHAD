.. index:: pair: struct; expo_inform_type
.. _doxid-structexpo__inform__type:

expo_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct expo_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          bad_eval::NTuple{13,Cchar}
          iter::INT
          cg_iter::INT
          fc_eval::INT
          gj_eval::INT
          hl_eval::INT
          obj::T
          primal_infeasibility::T
          dual_infeasibility::T
          complementary_slackness::T
          time::expo_time_type{T}
          bsc_inform::bsc_inform_type{T,INT}
          tru_inform::tru_inform_type{T,INT}
          ssls_inform::ssls_inform_type{T,INT}

.. _details-structexpo__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structexpo__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See EXPO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structexpo__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structexpo__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structexpo__inform__type_bad_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structexpo__inform__type_iter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; fc_eval
.. _doxid-structexpo__inform__type_fc_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT fc_eval

the total number of evaluations of the objective $f(x)$ and constraint $c(x)$ functions

.. index:: pair: variable; gj_eval
.. _doxid-structexpo__inform__type_gj_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT gj_eval

the total number of evaluations of the gradient $g(x)$ and Jacobian $J(x)$

.. index:: pair: variable; hl_eval
.. _doxid-structexpo__inform__type_hl_eval:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT hl_eval

the total number of evaluations of the Hessian $H(x,y)$ of the Lagrangian


.. index:: pair: variable; obj
.. _doxid-structexpo__inform__type_obj:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function $f(x)$ at the best estimate the solution, x, determined by EXPO_solve

.. index:: pair: variable; primal_infeasibility
.. _doxid-structexpo__inform__type_primal_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T primal_infeasibility

the norm of the primal infeasibility (1) at the best estimate of the solution x, determined by EXPO_solve

.. index:: pair: variable; dual_infeasibility
.. _doxid-structexpo__inform__type_dual_infeasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dual_infeasibility

the norm of the dual infeasibility (2) at the best estimate of the solution (x,y,z), determined by EXPO_solve

.. index:: pair: variable; complementary_slacknes
.. _doxid-structexpo__inform__type_complementary_slacknes:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T complementary_slackness

the norm of the complementary slackness (3) at the best estimate of the solution (x,y,z), determined by EXPO_solve

.. index:: pair: variable; time
.. _doxid-structexpo__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`expo_time_type<doxid-structexpo__time__type>` time

timings (see above)

.. index:: pair: variable; bsc_inform
.. _doxid-structexpo__inform__type_bsc_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` bsc_inform

inform parameters for BSC

.. index:: pair: variable; tru_inform
.. _doxid-structexpo__inform__type_tru_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`tru_inform_type<doxid-structtru__inform__type>` tru_inform

inform parameters for TRU

.. index:: pair: variable; ssls_inform
.. _doxid-structexpo__inform__type_ssls_inform:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ssls_inform_type<doxid-structssls__inform__type>` ssls_inform

inform parameters for SSLS
