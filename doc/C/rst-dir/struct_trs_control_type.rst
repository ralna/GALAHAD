.. index:: pair: table; trs_control_type
.. _doxid-structtrs__control__type:

trs_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_trs.h>
	
	struct trs_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structtrs__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structtrs__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structtrs__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`problem<doxid-structtrs__control__type_problem>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structtrs__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dense_factorization<doxid-structtrs__control__type_dense_factorization>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_h<doxid-structtrs__control__type_new_h>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_m<doxid-structtrs__control__type_new_m>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`new_a<doxid-structtrs__control__type_new_a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_factorizations<doxid-structtrs__control__type_max_factorizations>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`inverse_itmax<doxid-structtrs__control__type_inverse_itmax>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`taylor_max_degree<doxid-structtrs__control__type_taylor_max_degree>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_multiplier<doxid-structtrs__control__type_initial_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lower<doxid-structtrs__control__type_lower>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`upper<doxid-structtrs__control__type_upper>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_normal<doxid-structtrs__control__type_stop_normal>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute_normal<doxid-structtrs__control__type_stop_absolute_normal>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_hard<doxid-structtrs__control__type_stop_hard>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`start_invit_tol<doxid-structtrs__control__type_start_invit_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`start_invitmax_tol<doxid-structtrs__control__type_start_invitmax_tol>`;
		bool :ref:`equality_problem<doxid-structtrs__control__type_equality_problem>`;
		bool :ref:`use_initial_multiplier<doxid-structtrs__control__type_use_initial_multiplier>`;
		bool :ref:`initialize_approx_eigenvector<doxid-structtrs__control__type_initialize_approx_eigenvector>`;
		bool :ref:`force_Newton<doxid-structtrs__control__type_force_Newton>`;
		bool :ref:`space_critical<doxid-structtrs__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structtrs__control__type_deallocate_error_fatal>`;
		char :ref:`problem_file<doxid-structtrs__control__type_problem_file>`[31];
		char :ref:`symmetric_linear_solver<doxid-structtrs__control__type_symmetric_linear_solver>`[31];
		char :ref:`definite_linear_solver<doxid-structtrs__control__type_definite_linear_solver>`[31];
		char :ref:`prefix<doxid-structtrs__control__type_prefix>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structtrs__control__type_sls_control>`;
		struct :ref:`ir_control_type<doxid-structir__control__type>` :ref:`ir_control<doxid-structtrs__control__type_ir_control>`;
	};
.. _details-structtrs__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtrs__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtrs__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structtrs__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; problem
.. _doxid-structtrs__control__type_problem:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` problem

unit to write problem data into file problem_file

.. index:: pair: variable; print_level
.. _doxid-structtrs__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; dense_factorization
.. _doxid-structtrs__control__type_dense_factorization:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dense_factorization

should the problem be solved by dense factorization? Possible values are

* 0 sparse factorization will be used

* 1 dense factorization will be used

* other the choice is made automatically depending on the dimension & sparsity

.. index:: pair: variable; new_h
.. _doxid-structtrs__control__type_new_h:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_h

how much of $H$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_m
.. _doxid-structtrs__control__type_new_m:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_m

how much of $M$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_a
.. _doxid-structtrs__control__type_new_a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` new_a

how much of $A$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; max_factorizations
.. _doxid-structtrs__control__type_max_factorizations:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_factorizations

the maximum number of factorizations (=iterations) allowed. -ve implies no limit

.. index:: pair: variable; inverse_itmax
.. _doxid-structtrs__control__type_inverse_itmax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` inverse_itmax

the number of inverse iterations performed in the "maybe hard" case

.. index:: pair: variable; taylor_max_degree
.. _doxid-structtrs__control__type_taylor_max_degree:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` taylor_max_degree

maximum degree of Taylor approximant allowed

.. index:: pair: variable; initial_multiplier
.. _doxid-structtrs__control__type_initial_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_multiplier

initial estimate of the Lagrange multipler

.. index:: pair: variable; lower
.. _doxid-structtrs__control__type_lower:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structtrs__control__type_upper:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structtrs__control__type_stop_normal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_normal

stop when $| ||x|| - radius | \leq$ max( stop_normal \* radius, stop_absolute_normal )

.. index:: pair: variable; stop_absolute_normal
.. _doxid-structtrs__control__type_stop_absolute_normal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute_normal

see stop_normal

.. index:: pair: variable; stop_hard
.. _doxid-structtrs__control__type_stop_hard:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_hard

stop when bracket on optimal multiplier $\leq$ stop_hard \* max( bracket ends )

.. index:: pair: variable; start_invit_tol
.. _doxid-structtrs__control__type_start_invit_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` start_invit_tol

start inverse iteration when bracket on optimal multiplier $\leq$ stop_start_invit_tol \* max( bracket ends )

.. index:: pair: variable; start_invitmax_tol
.. _doxid-structtrs__control__type_start_invitmax_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` start_invitmax_tol

start full inverse iteration when bracket on multiplier $\leq$ stop_start_invitmax_tol \* max( bracket ends)

.. index:: pair: variable; equality_problem
.. _doxid-structtrs__control__type_equality_problem:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool equality_problem

is the solution is <b<required to lie on the boundary (i.e., is the constraint an equality)?

.. index:: pair: variable; use_initial_multiplier
.. _doxid-structtrs__control__type_use_initial_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool use_initial_multiplier

ignore initial_multiplier?

.. index:: pair: variable; initialize_approx_eigenvector
.. _doxid-structtrs__control__type_initialize_approx_eigenvector:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool initialize_approx_eigenvector

should a suitable initial eigenvector should be chosen or should a previous eigenvector may be used?

.. index:: pair: variable; force_Newton
.. _doxid-structtrs__control__type_force_Newton:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool force_Newton

ignore the trust-region if $H$ is positive definite

.. index:: pair: variable; space_critical
.. _doxid-structtrs__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtrs__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; problem_file
.. _doxid-structtrs__control__type_problem_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char problem_file[31]

name of file into which to write problem data

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structtrs__control__type_symmetric_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; definite_linear_solver
.. _doxid-structtrs__control__type_definite_linear_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr', 'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structtrs__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structtrs__control__type_sls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the Cholesky factorization and solution (see sls_c documentation)

.. index:: pair: variable; ir_control
.. _doxid-structtrs__control__type_ir_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement (see ir_c documentation)

