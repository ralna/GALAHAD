.. index:: pair: struct; cro_control_type
.. _doxid-structcro__control__type:

cro_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_cro.h>
	
	struct cro_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structcro__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structcro__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structcro__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structcro__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_schur_complement<doxid-structcro__control__type_max_schur_complement>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structcro__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`feasibility_tolerance<doxid-structcro__control__type_feasibility_tolerance>`;
		bool :ref:`check_io<doxid-structcro__control__type_check_io>`;
		bool :ref:`refine_solution<doxid-structcro__control__type_refine_solution>`;
		bool :ref:`space_critical<doxid-structcro__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structcro__control__type_deallocate_error_fatal>`;
		char :ref:`symmetric_linear_solver<doxid-structcro__control__type_symmetric_linear_solver>`[31];
		char :ref:`unsymmetric_linear_solver<doxid-structcro__control__type_unsymmetric_linear_solver>`[31];
		char :ref:`prefix<doxid-structcro__control__type_prefix>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structcro__control__type_sls_control>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structcro__control__type_sbls_control>`;
		struct :ref:`uls_control_type<doxid-structuls__control__type>` :ref:`uls_control<doxid-structcro__control__type_uls_control>`;
		struct :ref:`ir_control_type<doxid-structir__control__type>` :ref:`ir_control<doxid-structcro__control__type_ir_control>`;
	};
.. _details-structcro__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structcro__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structcro__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structcro__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structcro__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; max_schur_complement
.. _doxid-structcro__control__type_max_schur_complement:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_schur_complement

the maximum permitted size of the Schur complement before a refactorization is performed

.. index:: pair: variable; infinity
.. _doxid-structcro__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; feasibility_tolerance
.. _doxid-structcro__control__type_feasibility_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` feasibility_tolerance

feasibility tolerance for KKT violation

.. index:: pair: variable; check_io
.. _doxid-structcro__control__type_check_io:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool check_io

if .check_io is true, the input (x,y,z) will be fully tested for consistency

.. index:: pair: variable; refine_solution
.. _doxid-structcro__control__type_refine_solution:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool refine_solution

if .refine solution is true, attempt to satisfy the KKT conditions as accurately as possible

.. index:: pair: variable; space_critical
.. _doxid-structcro__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structcro__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structcro__control__type_symmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structcro__control__type_unsymmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

the name of the unsymmetric linear equation solver used. Possible choices are currently: 'gls', 'ma48' and 'getr', although only 'getr' is installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_uls<details-uls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structcro__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structcro__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; sbls_control
.. _doxid-structcro__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; uls_control
.. _doxid-structcro__control__type_uls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_control_type<doxid-structuls__control__type>` uls_control

control parameters for ULS

.. index:: pair: variable; ir_control
.. _doxid-structcro__control__type_ir_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement

