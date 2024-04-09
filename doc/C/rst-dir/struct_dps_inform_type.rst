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
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structdps__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structdps__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mod_1by1<doxid-structdps__inform__type_1a2a214743335ff43abbc71e6b00a50ea2>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mod_2by2<doxid-structdps__inform__type_1a6597a7caccf77ae67ed2b86f9281804a>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structdps__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_regularized<doxid-structdps__inform__type_1a1631e243108715d623e2ddb83310fa33>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`x_norm<doxid-structdps__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structdps__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pole<doxid-structdps__inform__type_1ad2dc9016b1d2b00a970ec28129f7000d>`;
		bool :ref:`hard_case<doxid-structdps__inform__type_1a22215075b7081ccac9f121daf07a0f7e>`;
		char :ref:`bad_alloc<doxid-structdps__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		struct :ref:`dps_time_type<doxid-structdps__time__type>` :ref:`time<doxid-structdps__inform__type_1ac1ae6841280bc6d2f24f9665cb75c40e>`;
		struct :ref:`sls_inform_type<doxid-structsls__inform__type>` :ref:`sls_inform<doxid-structdps__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`;
	};
.. _details-structdps__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structdps__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See DPS_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structdps__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; mod_1by1
.. _doxid-structdps__inform__type_1a2a214743335ff43abbc71e6b00a50ea2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mod_1by1

the number of 1 by 1 blocks from the factorization of H that were modified when constructing $M$

.. index:: pair: variable; mod_2by2
.. _doxid-structdps__inform__type_1a6597a7caccf77ae67ed2b86f9281804a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mod_2by2

the number of 2 by 2 blocks from the factorization of H that were modified when constructing $M$

.. index:: pair: variable; obj
.. _doxid-structdps__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structdps__inform__type_1a1631e243108715d623e2ddb83310fa33:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structdps__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` x_norm

the M-norm of the solution

.. index:: pair: variable; multiplier
.. _doxid-structdps__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the Lagrange multiplier associated with the constraint/regularization

.. index:: pair: variable; pole
.. _doxid-structdps__inform__type_1ad2dc9016b1d2b00a970ec28129f7000d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pole

a lower bound max(0,-lambda_1), where lambda_1 is the left-most eigenvalue of $(H,M)$

.. index:: pair: variable; hard_case
.. _doxid-structdps__inform__type_1a22215075b7081ccac9f121daf07a0f7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structdps__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structdps__inform__type_1ac1ae6841280bc6d2f24f9665cb75c40e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`dps_time_type<doxid-structdps__time__type>` time

time information

.. index:: pair: variable; sls_inform
.. _doxid-structdps__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_inform_type<doxid-structsls__inform__type>` sls_inform

information from SLS

