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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structcro__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structcro__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structcro__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dependent<doxid-structcro__inform__type_1a3678dbffc0e2f3521f7ef27194b21ab6>`;
		struct :ref:`cro_time_type<doxid-structcro__time__type>` :ref:`time<doxid-structcro__inform__type_1a0d99b2a30c1bf487fddf2643b03a3120>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structcro__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structcro__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
		struct :ref:`uls_inform_type<doxid-structuls__inform__type>` :ref:`uls_inform<doxid-structcro__inform__type_1aa39eb0d7b50d4a858849f8ef652ae84c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scu_status<doxid-structcro__inform__type_1a25bf1e7f86c2b4f4836aa4de40019815>`;
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>` :ref:`scu_inform<doxid-structcro__inform__type_1a0b702af94f05b9d4bb2bb6416f2498ee>`;
		struct :ref:`ir_inform_type<doxid-structir__inform__type>` :ref:`ir_inform<doxid-structcro__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5>`;
	};
.. _details-structcro__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structcro__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See CRO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structcro__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structcro__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; dependent
.. _doxid-structcro__inform__type_1a3678dbffc0e2f3521f7ef27194b21ab6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dependent

the number of dependent active constraints

.. index:: pair: variable; time
.. _doxid-structcro__inform__type_1a0d99b2a30c1bf487fddf2643b03a3120:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`cro_time_type<doxid-structcro__time__type>` time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structcro__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structcro__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

information from SBLS

.. index:: pair: variable; uls_inform
.. _doxid-structcro__inform__type_1aa39eb0d7b50d4a858849f8ef652ae84c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_inform_type<doxid-structuls__inform__type>` uls_inform

information from ULS

.. index:: pair: variable; scu_status
.. _doxid-structcro__inform__type_1a25bf1e7f86c2b4f4836aa4de40019815:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scu_status

information from SCU

.. index:: pair: variable; scu_inform
.. _doxid-structcro__inform__type_1a0b702af94f05b9d4bb2bb6416f2498ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`scu_inform_type<doxid-structscu__inform__type>` scu_inform

see scu_status

.. index:: pair: variable; ir_inform
.. _doxid-structcro__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_inform_type<doxid-structir__inform__type>` ir_inform

information from IR

