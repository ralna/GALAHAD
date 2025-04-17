.. index:: pair: table; llsr_control_type
.. _doxid-structllsr__control__type:

llsr_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_llsr.h>
	
	struct llsr_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structllsr__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structllsr__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structllsr__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structllsr__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_a<doxid-structllsr__control__type_new_a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_s<doxid-structllsr__control__type_new_s>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_factorizations<doxid-structllsr__control__type_max_factorizations>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`taylor_max_degree<doxid-structllsr__control__type_taylor_max_degree>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_multiplier<doxid-structllsr__control__type_initial_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lower<doxid-structllsr__control__type_lower>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`upper<doxid-structllsr__control__type_upper>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_normal<doxid-structllsr__control__type_stop_normal>`;
		bool :ref:`use_initial_multiplier<doxid-structllsr__control__type_use_initial_multiplier>`;
		bool :ref:`space_critical<doxid-structllsr__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structllsr__control__type_deallocate_error_fatal>`;
		char :ref:`definite_linear_solver<doxid-structllsr__control__type_definite_linear_solver>`[31];
		char :ref:`prefix<doxid-structllsr__control__type_prefix>`[31];
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structllsr__control__type_sbls_control>`;
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structllsr__control__type_sls_control>`;
		struct :ref:`ir_control_type<doxid-structir__control__type>` :ref:`ir_control<doxid-structllsr__control__type_ir_control>`;
	};
.. _details-structllsr__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structllsr__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structllsr__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structllsr__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structllsr__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; new_a
.. _doxid-structllsr__control__type_new_a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_a

how much of $A$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_s
.. _doxid-structllsr__control__type_new_s:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_s

how much of $S$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; max_factorizations
.. _doxid-structllsr__control__type_max_factorizations:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_factorizations

the maximum number of factorizations (=iterations) allowed. -ve implies no limit

.. index:: pair: variable; taylor_max_degree
.. _doxid-structllsr__control__type_taylor_max_degree:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` taylor_max_degree

maximum degree of Taylor approximant allowed (<= 3)

.. index:: pair: variable; initial_multiplier
.. _doxid-structllsr__control__type_initial_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_multiplier

initial estimate of the Lagrange multipler

.. index:: pair: variable; lower
.. _doxid-structllsr__control__type_lower:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structllsr__control__type_upper:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structllsr__control__type_stop_normal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_normal

stop when $| \|x\| - (\lambda/\sigma)^{1/(p-2)| \leq$ stop_normal \* max( 1, $\|x\|$)

.. index:: pair: variable; use_initial_multiplier
.. _doxid-structllsr__control__type_use_initial_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool use_initial_multiplier

ignore initial_multiplier?

.. index:: pair: variable; space_critical
.. _doxid-structllsr__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structllsr__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; definite_linear_solver
.. _doxid-structllsr__control__type_definite_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr',  'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structllsr__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structllsr__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_control
.. _doxid-structllsr__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_control
.. _doxid-structllsr__control__type_ir_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement for definite system solves (see ir_c documentation)

