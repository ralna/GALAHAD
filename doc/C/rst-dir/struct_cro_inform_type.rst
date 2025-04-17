.. index:: pair: struct; cro_inform_type
.. _doxid-structcro__inform__type:

cro_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_cro.h>
	
	struct cro_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structcro__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structcro__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structcro__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dependent<doxid-structcro__inform__type_dependent>`;
		struct :ref:`cro_time_type<doxid-structcro__time__type>` :ref:`time<doxid-structcro__inform__type_time>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structcro__inform__type_sls_inform>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structcro__inform__type_sbls_inform>`;
		struct :ref:`uls_inform_type<doxid-structuls__inform__type>` :ref:`uls_inform<doxid-structcro__inform__type_uls_inform>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scu_status<doxid-structcro__inform__type_scu_status>`;
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>` :ref:`scu_inform<doxid-structcro__inform__type_scu_inform>`;
		struct :ref:`ir_inform_type<doxid-structir__inform__type>` :ref:`ir_inform<doxid-structcro__inform__type_ir_inform>`;
	};
.. _details-structcro__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structcro__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See CRO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structcro__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structcro__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; dependent
.. _doxid-structcro__inform__type_dependent:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dependent

the number of dependent active constraints

.. index:: pair: variable; time
.. _doxid-structcro__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`cro_time_type<doxid-structcro__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structcro__inform__type_sls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structcro__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from SBLS

.. index:: pair: variable; uls_inform
.. _doxid-structcro__inform__type_uls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

information from ULS

.. index:: pair: variable; scu_status
.. _doxid-structcro__inform__type_scu_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scu_status

information from SCU

.. index:: pair: variable; scu_inform
.. _doxid-structcro__inform__type_scu_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`scu_inform_type<doxid-structscu__inform__type>` scu_inform

see scu_status

.. index:: pair: variable; ir_inform
.. _doxid-structcro__inform__type_ir_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from IR

