.. index:: pair: struct; slls_inform_type
.. _doxid-structslls__inform__type:

slls_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_slls.h>
	
	struct slls_inform_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structslls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structslls__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorization_status<doxid-structslls__inform__type_factorization_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structslls__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_iter<doxid-structslls__inform__type_cg_iter>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structslls__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`norm_pg<doxid-structslls__inform__type_norm_pg>`;
		char :ref:`bad_alloc<doxid-structslls__inform__type_bad_alloc>`[81];
		struct :ref:`slls_time_type<doxid-structslls__time__type>` :ref:`time<doxid-structslls__inform__type_time>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structslls__inform__type_sbls_inform>`;
		struct :ref:`convert_inform_type<doxid-structconvert__inform__type>` :ref:`convert_inform<doxid-structslls__inform__type_convert_inform>`;
	};
.. _details-structslls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structslls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

reported return status.

.. index:: pair: variable; alloc_status
.. _doxid-structslls__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

Fortran STAT value after allocate failure.

.. index:: pair: variable; factorization_status
.. _doxid-structslls__inform__type_factorization_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorization_status

status return from factorization

.. index:: pair: variable; iter
.. _doxid-structslls__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

number of iterations required

.. index:: pair: variable; cg_iter
.. _doxid-structslls__inform__type_cg_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_iter

number of CG iterations required

.. index:: pair: variable; obj
.. _doxid-structslls__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

current value of the objective function, r(x).

.. index:: pair: variable; norm_pg
.. _doxid-structslls__inform__type_norm_pg:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` norm_pg

current value of the Euclidean norm of projected gradient of r(x).

.. index:: pair: variable; bad_alloc
.. _doxid-structslls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structslls__inform__type_time:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`slls_time_type<doxid-structslls__time__type>` time

times for various stages

.. index:: pair: variable; sbls_inform
.. _doxid-structslls__inform__type_sbls_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform values from SBLS

.. index:: pair: variable; convert_inform
.. _doxid-structslls__inform__type_convert_inform:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`convert_inform_type<doxid-structconvert__inform__type>` convert_inform

inform values for CONVERT

