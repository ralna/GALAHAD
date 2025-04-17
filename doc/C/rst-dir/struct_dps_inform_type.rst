.. index:: pair: table; dps_inform_type
.. _doxid-structdps__inform__type:

dps_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dps.h>
	
	struct dps_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structdps__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structdps__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mod_1by1<doxid-structdps__inform__type_mod_1by1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mod_2by2<doxid-structdps__inform__type_mod_2by2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structdps__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_regularized<doxid-structdps__inform__type_obj_regularized>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structdps__inform__type_x_norm>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structdps__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pole<doxid-structdps__inform__type_pole>`;
		bool :ref:`hard_case<doxid-structdps__inform__type_hard_case>`;
		char :ref:`bad_alloc<doxid-structdps__inform__type_bad_alloc>`[81];
		struct :ref:`dps_time_type<doxid-structdps__time__type>` :ref:`time<doxid-structdps__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structdps__inform__type_sls_inform>`;
	};
.. _details-structdps__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdps__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See DPS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdps__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; mod_1by1
.. _doxid-structdps__inform__type_mod_1by1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mod_1by1

the number of 1 by 1 blocks from the factorization of H that were modified when constructing $M$

.. index:: pair: variable; mod_2by2
.. _doxid-structdps__inform__type_mod_2by2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mod_2by2

the number of 2 by 2 blocks from the factorization of H that were modified when constructing $M$

.. index:: pair: variable; obj
.. _doxid-structdps__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structdps__inform__type_obj_regularized:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structdps__inform__type_x_norm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the M-norm of the solution

.. index:: pair: variable; multiplier
.. _doxid-structdps__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the Lagrange multiplier associated with the constraint/regularization

.. index:: pair: variable; pole
.. _doxid-structdps__inform__type_pole:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pole

a lower bound max(0,-lambda_1), where lambda_1 is the left-most eigenvalue of $(H,M)$

.. index:: pair: variable; hard_case
.. _doxid-structdps__inform__type_hard_case:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structdps__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structdps__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dps_time_type<doxid-structdps__time__type>` time

time information

.. index:: pair: variable; sls_inform
.. _doxid-structdps__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from SLS

