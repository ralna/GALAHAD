.. index:: pair: table; ssls_control_type
.. _doxid-structssls__control__type:

ssls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ssls.h>
	
	struct ssls_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structssls__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structssls__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structssls__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structssls__control__type_print_level>`;
		bool :ref:`space_critical<doxid-structssls__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structssls__control__type_deallocate_error_fatal>`;
		char :ref:`symmetric_linear_solver<doxid-structssls__control__type_symmetric_linear_solver>`[31];
		char :ref:`prefix<doxid-structssls__control__type_prefix>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structssls__control__type_sls_control>`;
	};
.. _details-structssls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structssls__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structssls__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structssls__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structssls__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; space_critical
.. _doxid-structssls__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structssls__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structssls__control__type_symmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structssls__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structssls__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

