.. index:: pair: table; sec_control_type
.. _doxid-structsec__control__type:

sec_control_type structure
--------------------------

.. toctree::
	:hidden:

.. _details-structsec__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: struct; sec_control_type
.. _doxid-structsec__control__type:

struct sec_control_type
=======================

.. toctree::
	:hidden:

Overview
~~~~~~~~

control derived type as a C struct :ref:`More...<details-structsec__control__type>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sec.h>
	
	struct sec_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structsec__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		int :ref:`error<doxid-structsec__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		int :ref:`out<doxid-structsec__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		int :ref:`print_level<doxid-structsec__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`h_initial<doxid-structsec__control__type_1a023bd6b7e060144782755238a1da549e>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`update_skip_tol<doxid-structsec__control__type_1a8dfc46d0fb22a5d3b62f751e8c4a024b>`;
		char :ref:`prefix<doxid-structsec__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structsec__control__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

control derived type as a C struct

Fields
------

.. index:: pair: variable; f_indexing
.. _doxid-structsec__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsec__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsec__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsec__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int print_level

the level of output required. <= 0 gives no output, >= 1 warning message

.. index:: pair: variable; h_initial
.. _doxid-structsec__control__type_1a023bd6b7e060144782755238a1da549e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` h_initial

the initial Hessian approximation will be h_initial \* :math:`I`

.. index:: pair: variable; update_skip_tol
.. _doxid-structsec__control__type_1a8dfc46d0fb22a5d3b62f751e8c4a024b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` update_skip_tol

an update is skipped if the resulting matrix would have grown too much; specifically it is skipped when y^T s / y^T y <= update_skip_tol.

.. index:: pair: variable; prefix
.. _doxid-structsec__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

