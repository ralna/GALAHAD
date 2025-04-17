.. index:: pair: struct; dgo_control_type
.. _doxid-structdgo__control__type:

dgo_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dgo.h>
	
	struct dgo_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structdgo__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structdgo__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structdgo__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structdgo__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structdgo__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structdgo__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structdgo__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structdgo__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_evals<doxid-structdgo__control__type_max_evals>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dictionary_size<doxid-structdgo__control__type_dictionary_size>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structdgo__control__type_alive_unit>`;
		char :ref:`alive_file<doxid-structdgo__control__type_alive_file>`[31];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structdgo__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lipschitz_lower_bound<doxid-structdgo__control__type_lipschitz_lower_bound>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lipschitz_reliability<doxid-structdgo__control__type_lipschitz_reliability>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lipschitz_control<doxid-structdgo__control__type_lipschitz_control>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_length<doxid-structdgo__control__type_stop_length>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_f<doxid-structdgo__control__type_stop_f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structdgo__control__type_obj_unbounded>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structdgo__control__type_cpu_time_limit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structdgo__control__type_clock_time_limit>`;
		bool :ref:`hessian_available<doxid-structdgo__control__type_hessian_available>`;
		bool :ref:`prune<doxid-structdgo__control__type_prune>`;
		bool :ref:`perform_local_optimization<doxid-structdgo__control__type_perform_local_optimization>`;
		bool :ref:`space_critical<doxid-structdgo__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structdgo__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structdgo__control__type_prefix>`[31];
		struct :ref:`hash_control_type<doxid-structhash__control__type>` :ref:`hash_control<doxid-structdgo__control__type_hash_control>`;
		struct :ref:`ugo_control_type<doxid-structugo__control__type>` :ref:`ugo_control<doxid-structdgo__control__type_ugo_control>`;
		struct :ref:`trb_control_type<doxid-structtrb__control__type>` :ref:`trb_control<doxid-structdgo__control__type_trb_control>`;
	};
.. _details-structdgo__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structdgo__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structdgo__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structdgo__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structdgo__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* 1 a one-line summary for every improvement

* 2 a summary of each iteration

* $\geq$ 3 increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structdgo__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structdgo__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structdgo__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structdgo__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations performed

.. index:: pair: variable; max_evals
.. _doxid-structdgo__control__type_max_evals:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_evals

the maximum number of function evaluations made

.. index:: pair: variable; dictionary_size
.. _doxid-structdgo__control__type_dictionary_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dictionary_size

the size of the initial hash dictionary

.. index:: pair: variable; alive_unit
.. _doxid-structdgo__control__type_alive_unit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structdgo__control__type_alive_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; infinity
.. _doxid-structdgo__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; lipschitz_lower_bound
.. _doxid-structdgo__control__type_lipschitz_lower_bound:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lipschitz_lower_bound

a small positive constant (<= 1e-6) that ensure that the estimted gradient Lipschitz constant is not too small

.. index:: pair: variable; lipschitz_reliability
.. _doxid-structdgo__control__type_lipschitz_reliability:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lipschitz_reliability

the Lipschitz reliability parameter, the Lipschiz constant used will be a factor lipschitz_reliability times the largest value observed

.. index:: pair: variable; lipschitz_control
.. _doxid-structdgo__control__type_lipschitz_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lipschitz_control

the reliablity control parameter, the actual reliability parameter used will be .lipschitz_reliability

* MAX( 1, n - 1 ) \* .lipschitz_control / iteration

.. index:: pair: variable; stop_length
.. _doxid-structdgo__control__type_stop_length:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_length

the iteration will stop if the length, delta, of the diagonal in the box with the smallest-found objective function is smaller than .stop_length times that of the original bound box, delta_0

.. index:: pair: variable; stop_f
.. _doxid-structdgo__control__type_stop_f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_f

the iteration will stop if the gap between the best objective value found and the smallest lower bound is smaller than .stop_f

.. index:: pair: variable; obj_unbounded
.. _doxid-structdgo__control__type_obj_unbounded:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structdgo__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structdgo__control__type_clock_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structdgo__control__type_hessian_available:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; prune
.. _doxid-structdgo__control__type_prune:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool prune

should boxes that cannot contain the global minimizer be pruned (i.e., removed from further consideration)?

.. index:: pair: variable; perform_local_optimization
.. _doxid-structdgo__control__type_perform_local_optimization:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool perform_local_optimization

should approximate minimizers be impoved by judicious local minimization?

.. index:: pair: variable; space_critical
.. _doxid-structdgo__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structdgo__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structdgo__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; hash_control
.. _doxid-structdgo__control__type_hash_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`hash_control_type<doxid-structhash__control__type>` hash_control

control parameters for HASH

.. index:: pair: variable; ugo_control
.. _doxid-structdgo__control__type_ugo_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ugo_control_type<doxid-structugo__control__type>` ugo_control

control parameters for UGO

.. index:: pair: variable; trb_control
.. _doxid-structdgo__control__type_trb_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trb_control_type<doxid-structtrb__control__type>` trb_control

control parameters for TRB

