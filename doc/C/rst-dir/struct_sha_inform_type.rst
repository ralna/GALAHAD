.. index:: pair: table; sha_inform_type
.. _doxid-structsha__inform__type:

sha_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sha.h>

	struct sha_inform_type {
		// fields

		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structsha__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structsha__inform__type_alloc_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_degree<doxid-structsha__inform__type_max_degree>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`differences_needed<doxid-structsha__inform__type_differences_needed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_reduced_degree<doxid-structsha__inform__type_max_reduced_degree>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`approximation_algorithm_used<doxid-structsha__inform__type_approximation_algorithm_used>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`bad_row<doxid-structsha__inform__type_bad_row>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_off_diagonal_difference<doxid-structsha__inform__type_1a42d0c89df887685f68327d07c6e92f04>`;
		char :ref:`bad_alloc<doxid-structsha__inform__type_bad_alloc>`[81];
	};
.. _details-structsha__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsha__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See SHA_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structsha__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; max_degree
.. _doxid-structsha__inform__type_max_degree:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_degree

the maximum degree in the adgacency graph.

.. index:: pair: variable; differences_needed
.. _doxid-structsha__inform__type_differences_needed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` differences_needed

the number of differences that will be needed.

.. index:: pair: variable; max_reduced_degree
.. _doxid-structsha__inform__type_max_reduced_degree:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_reduced_degree

the maximum reduced degree in the adgacency graph.

.. index:: pair: variable; approximation_algorithm_used
.. _doxid-structsha__inform__type_approximation_algorithm_used:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` approximation_algorithm_used

the approximation algorithm actually used

.. index:: pair: variable; bad_row
.. _doxid-structsha__inform__type_bad_row:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` bad_row

a failure occured when forming the bad_row-th row (0 = no failure).


.. index:: pair: variable; max_off_diagonal_difference
.. _doxid-structsha__inform__type_1a42d0c89df887685f68327d07c6e92f04:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__rpc_8h_>` max_off_diagonal_difference

the maximum difference between estimated Hessian off-diagonal 
pairs if approximation algorithm 1, 3 or 4 has been employed.

.. index:: pair: variable; bad_alloc
.. _doxid-structsha__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred.
