.. index:: pair: struct; expo_inform_type
.. _doxid-structexpo__inform__type:

expo_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_expo.h>
	
	struct expo_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structexpo__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structexpo__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structexpo__inform__type_bad_alloc>`[81];
		char :ref:`bad_eval<doxid-structexpo__inform__type_bad_eval>`[13];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structexpo__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`fc_eval<doxid-structexpo__inform__type_fc_eval>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`gj_eval<doxid-structexpo__inform__type_gj_eval>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structexpo__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`primal_infeasibility<doxid-structexpo__inform__type_primal_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dual_infeasibility<doxid-structexpo__inform__type_dual_infeasibility>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`complementary_slackness<doxid-structexpo__inform__type_complementary_slackness>`;
		struct :ref:`expo_time_type<doxid-structexpo__time__type>` :ref:`time<doxid-structexpo__inform__type_time>`;
		struct :ref:`tru_inform_type<doxid-structtru__inform__type>` :ref:`tru_inform<doxid-structexpo__inform__type_tru_inform>`;
		struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` :ref:`bsc_inform<doxid-structexpo__inform__type_bsc_inform>`;
		struct :ref:`ssls_inform_type<doxid-structssls__inform__type>` :ref:`ssls_inform<doxid-structexpo__inform__type_ssls_inform>`;
	};
.. _details-structexpo__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structexpo__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See EXPO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structexpo__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structexpo__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; bad_eval
.. _doxid-structexpo__inform__type_bad_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_eval[13]

the name of the user-supplied evaluation routine for which an error occurred

.. index:: pair: variable; iter
.. _doxid-structexpo__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations performed

.. index:: pair: variable; fc_eval
.. _doxid-structexpo__inform__type_fc_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` fc_eval

the total number of evaluations of the objective f(x) and constraints c(x)

.. index:: pair: variable; gj_eval
.. _doxid-structexpo__inform__type_gj_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` gj_eval

the total number of evaluations of the gradient g(x) of f(x) and Jacobian J(x) of c(x)

.. index:: pair: variable; hl_eval
.. _doxid-structexpo__inform__type_hl_eval:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` hl_eval

the total number of evaluations of the Hessian H(x,y) of the Lagrangian

.. index:: pair: variable; obj
.. _doxid-structexpo__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the objective function $f(x)$ at the best estimate the solution, x, determined by EXPO_solve

.. index:: pair: variable;  primal_infeasibility
.. _doxid-structexpo__inform__type_primal_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` primal_infeasibility

the norm of the primal infeasibility (1) at the best estimate of the solution x, determined by EXPO_solve

.. index:: pair: variable; dual_infeasibility
.. _doxid-structexpo__inform__type_dual_infeasibility:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dual_infeasibility

the norm of the dual infeasibility (2) at the best estimate, (x,y,z), of the solution determined by EXPO_solve

.. index:: pair: variable; complementary_slackness
.. _doxid-structexpo__inform__type_complementary_slackness:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` complementary_slackness

the norm of the complementary slackness (3) at the best estimate, (x,y,z), of the solution determined by EXPO_solve

.. index:: pair: variable; time
.. _doxid-structexpo__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`expo_time_type<doxid-structexpo__time__type>` time

timings (see above)

.. index:: pair: variable; bsc_inform
.. _doxid-structexpo__inform__type_bsc_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`bsc_inform_type<doxid-structbsc__inform__type>` bsc_inform

inform parameters for BSC

.. index:: pair: variable; tru_inform
.. _doxid-structexpo__inform__type_tru_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`tru_inform_type<doxid-structtru__inform__type>` tru_inform

inform parameters for TRU

.. index:: pair: variable; ssls_inform
.. _doxid-structexpo__inform__type_ssls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ssls_inform_type<doxid-structssls__inform__type>` ssls_inform

inform parameters for SSLS


