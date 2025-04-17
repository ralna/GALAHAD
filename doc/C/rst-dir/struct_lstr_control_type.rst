.. index:: pair: table; lstr_control_type
.. _doxid-structlstr__control__type:

lstr_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lstr.h>
	
	struct lstr_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structlstr__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlstr__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlstr__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlstr__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structlstr__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structlstr__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structlstr__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmin<doxid-structlstr__control__type_itmin>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmax<doxid-structlstr__control__type_itmax>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmax_on_boundary<doxid-structlstr__control__type_itmax_on_boundary>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`bitmax<doxid-structlstr__control__type_bitmax>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_vectors<doxid-structlstr__control__type_extra_vectors>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_relative<doxid-structlstr__control__type_stop_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute<doxid-structlstr__control__type_stop_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`fraction_opt<doxid-structlstr__control__type_fraction_opt>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`time_limit<doxid-structlstr__control__type_time_limit>`;
		bool :ref:`steihaug_toint<doxid-structlstr__control__type_steihaug_toint>`;
		bool :ref:`space_critical<doxid-structlstr__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structlstr__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structlstr__control__type_prefix>`[31];
	};
.. _details-structlstr__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlstr__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlstr__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlstr__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlstr__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structlstr__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlstr__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structlstr__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; itmin
.. _doxid-structlstr__control__type_itmin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmin

the minimum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax
.. _doxid-structlstr__control__type_itmax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax_on_boundary
.. _doxid-structlstr__control__type_itmax_on_boundary:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmax_on_boundary

the maximum number of iterations allowed once the boundary has been encountered (-ve = no bound)

.. index:: pair: variable; bitmax
.. _doxid-structlstr__control__type_bitmax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` bitmax

the maximum number of Newton inner iterations per outer iteration allowe (-ve = no bound)

.. index:: pair: variable; extra_vectors
.. _doxid-structlstr__control__type_extra_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; stop_relative
.. _doxid-structlstr__control__type_stop_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_relative

the iteration stops successfully when $\|A^Tr\|$ is less than max( stop_relative \* $\|A^Tr_{initial} \|$, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structlstr__control__type_stop_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structlstr__control__type_fraction_opt:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; time_limit
.. _doxid-structlstr__control__type_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` time_limit

the maximum elapsed time allowed (-ve means infinite)

.. index:: pair: variable; steihaug_toint
.. _doxid-structlstr__control__type_steihaug_toint:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool steihaug_toint

should the iteration stop when the Trust-region is first encountered?

.. index:: pair: variable; space_critical
.. _doxid-structlstr__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlstr__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structlstr__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

