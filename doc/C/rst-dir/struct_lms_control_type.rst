.. index:: pair: table; lms_control_type
.. _doxid-structlms__control__type:

lms_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lms.h>
	
	struct lms_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structlms__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlms__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlms__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlms__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`memory_length<doxid-structlms__control__type_1a43017042f3cf20a8e9b364b4f7be0104>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`method<doxid-structlms__control__type_1adcc9a19ad3119f823a658f6a49a24e64>`;
		bool :ref:`any_method<doxid-structlms__control__type_1a1314157047f4fc989c8d680141a32d6b>`;
		bool :ref:`space_critical<doxid-structlms__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structlms__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structlms__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structlms__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlms__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlms__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structlms__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structlms__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; memory_length
.. _doxid-structlms__control__type_1a43017042f3cf20a8e9b364b4f7be0104:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` memory_length

limited memory length

.. index:: pair: variable; method
.. _doxid-structlms__control__type_1adcc9a19ad3119f823a658f6a49a24e64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` method

limited-memory formula required (others may be added in due course):

* 1 BFGS (default).

* 2 Symmetric Rank-One (SR1).

* 3 The inverse of the BFGS formula.

* 4 The inverse of the shifted BFGS formula. This should be used instead of .method = 3 whenever a shift is planned.

.. index:: pair: variable; any_method
.. _doxid-structlms__control__type_1a1314157047f4fc989c8d680141a32d6b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool any_method

allow space to permit different methods if required (less efficient)

.. index:: pair: variable; space_critical
.. _doxid-structlms__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlms__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structlms__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

