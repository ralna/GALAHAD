.. index:: pair: table; lsrt_control_type
.. _doxid-structlsrt__control__type:

lsrt_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lsrt.h>
	
	struct lsrt_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structlsrt__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlsrt__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlsrt__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlsrt__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structlsrt__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structlsrt__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structlsrt__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmin<doxid-structlsrt__control__type_itmin>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmax<doxid-structlsrt__control__type_itmax>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`bitmax<doxid-structlsrt__control__type_bitmax>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_vectors<doxid-structlsrt__control__type_extra_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stopping_rule<doxid-structlsrt__control__type_stopping_rule>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`freq<doxid-structlsrt__control__type_freq>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_relative<doxid-structlsrt__control__type_stop_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute<doxid-structlsrt__control__type_stop_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`fraction_opt<doxid-structlsrt__control__type_fraction_opt>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`time_limit<doxid-structlsrt__control__type_time_limit>`;
		bool :ref:`space_critical<doxid-structlsrt__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structlsrt__control__type_deallocate_error_fatal>`;
		char :ref:`prefix<doxid-structlsrt__control__type_prefix>`[31];
	};
.. _details-structlsrt__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlsrt__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlsrt__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlsrt__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlsrt__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structlsrt__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlsrt__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structlsrt__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; itmin
.. _doxid-structlsrt__control__type_itmin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmin

the minimum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax
.. _doxid-structlsrt__control__type_itmax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; bitmax
.. _doxid-structlsrt__control__type_bitmax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` bitmax

the maximum number of Newton inner iterations per outer iteration allowed (-ve = no bound)

.. index:: pair: variable; extra_vectors
.. _doxid-structlsrt__control__type_extra_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; stopping_rule
.. _doxid-structlsrt__control__type_stopping_rule:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stopping_rule

the stopping rule used: 0=1.0, 1=norm step, 2=norm step/sigma (NOT USED)

.. index:: pair: variable; freq
.. _doxid-structlsrt__control__type_freq:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` freq

frequency for solving the reduced tri-diagonal problem (NOT USED)

.. index:: pair: variable; stop_relative
.. _doxid-structlsrt__control__type_stop_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_relative

the iteration stops successfully when \|\|A^Tr\|\| is less than max( stop_relative \* \|\|A^Tr initial \|\|, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structlsrt__control__type_stop_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structlsrt__control__type_fraction_opt:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; time_limit
.. _doxid-structlsrt__control__type_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` time_limit

the maximum elapsed time allowed (-ve means infinite)

.. index:: pair: variable; space_critical
.. _doxid-structlsrt__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlsrt__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structlsrt__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

