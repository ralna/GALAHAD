.. index:: pair: table; sec_control_type
.. _doxid-structsec__control__type:

sec_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sec.h>
	
	struct sec_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structsec__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structsec__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structsec__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structsec__control__type_print_level>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`h_initial<doxid-structsec__control__type_h_initial>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`update_skip_tol<doxid-structsec__control__type_update_skip_tol>`;
		char :ref:`prefix<doxid-structsec__control__type_prefix>`[31];
	};
.. _details-structsec__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsec__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsec__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structsec__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structsec__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. <= 0 gives no output, >= 1 warning message

.. index:: pair: variable; h_initial
.. _doxid-structsec__control__type_h_initial:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` h_initial

the initial Hessian approximation will be h_initial \* $I$

.. index:: pair: variable; update_skip_tol
.. _doxid-structsec__control__type_update_skip_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` update_skip_tol

an update is skipped if the resulting matrix would have grown too much; specifically it is skipped when y^T s / y^T y <= update_skip_tol.

.. index:: pair: variable; prefix
.. _doxid-structsec__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

