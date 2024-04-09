.. index:: pair: table; convert_control_type
.. _doxid-structconvert__control__type:

convert_control_type structure
------------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_convert.h>
	
	struct convert_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structconvert__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structconvert__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structconvert__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structconvert__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		bool :ref:`transpose<doxid-structconvert__control__type_1a25a62de4b18dc349803bf5447052d673>`;
		bool :ref:`sum_duplicates<doxid-structconvert__control__type_1a6a46ec71c5b01b04c75c9bf0038d1762>`;
		bool :ref:`order<doxid-structconvert__control__type_1a0176c64e4b8b660bf4ac9cdc29b852ce>`;
		bool :ref:`space_critical<doxid-structconvert__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structconvert__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structconvert__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structconvert__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structconvert__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structconvert__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structconvert__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structconvert__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; transpose
.. _doxid-structconvert__control__type_1a25a62de4b18dc349803bf5447052d673:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool transpose

obtain the transpose of the input matrix?

.. index:: pair: variable; sum_duplicates
.. _doxid-structconvert__control__type_1a6a46ec71c5b01b04c75c9bf0038d1762:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool sum_duplicates

add the values of entries in duplicate positions?

.. index:: pair: variable; order
.. _doxid-structconvert__control__type_1a0176c64e4b8b660bf4ac9cdc29b852ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool order

order row or column data by increasing index?

.. index:: pair: variable; space_critical
.. _doxid-structconvert__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structconvert__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structconvert__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

