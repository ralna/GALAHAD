.. index:: pair: table; glrt_control_type
.. _doxid-structglrt__control__type:

glrt_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_glrt.h>
	
	struct glrt_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structglrt__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structglrt__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structglrt__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structglrt__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmax<doxid-structglrt__control__type_itmax>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stopping_rule<doxid-structglrt__control__type_stopping_rule>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`freq<doxid-structglrt__control__type_freq>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_vectors<doxid-structglrt__control__type_extra_vectors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ritz_printout_device<doxid-structglrt__control__type_ritz_printout_device>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_relative<doxid-structglrt__control__type_stop_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute<doxid-structglrt__control__type_stop_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`fraction_opt<doxid-structglrt__control__type_fraction_opt>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`rminvr_zero<doxid-structglrt__control__type_rminvr_zero>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`f_0<doxid-structglrt__control__type_f_0>`;
		bool :ref:`unitm<doxid-structglrt__control__type_unitm>`;
		bool :ref:`impose_descent<doxid-structglrt__control__type_impose_descent>`;
		bool :ref:`space_critical<doxid-structglrt__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structglrt__control__type_deallocate_error_fatal>`;
		bool :ref:`print_ritz_values<doxid-structglrt__control__type_print_ritz_values>`;
		char :ref:`ritz_file_name<doxid-structglrt__control__type_ritz_file_name>`[31];
		char :ref:`prefix<doxid-structglrt__control__type_prefix>`[31];
	};
.. _details-structglrt__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structglrt__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structglrt__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structglrt__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structglrt__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; itmax
.. _doxid-structglrt__control__type_itmax:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; stopping_rule
.. _doxid-structglrt__control__type_stopping_rule:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stopping_rule

the stopping rule used (see below). Possible values are:

* 1 stopping rule = norm of the step.

* 2 stopping rule is norm of the step / $\sigma$.

* other. stopping rule = 1.0.

.. index:: pair: variable; freq
.. _doxid-structglrt__control__type_freq:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` freq

frequency for solving the reduced tri-diagonal problem

.. index:: pair: variable; extra_vectors
.. _doxid-structglrt__control__type_extra_vectors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; ritz_printout_device
.. _doxid-structglrt__control__type_ritz_printout_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ritz_printout_device

the unit number for writing debug Ritz values

.. index:: pair: variable; stop_relative
.. _doxid-structglrt__control__type_stop_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_relative

the iteration stops successfully when the gradient in the $M^{-1}$ norm is smaller than max( stop_relative \* min( 1, stopping_rule ) \* norm initial gradient, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structglrt__control__type_stop_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structglrt__control__type_fraction_opt:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; rminvr_zero
.. _doxid-structglrt__control__type_rminvr_zero:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` rminvr_zero

the smallest value that the square of the M norm of the gradient of the objective may be before it is considered to be zero

.. index:: pair: variable; f_0
.. _doxid-structglrt__control__type_f_0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` f_0

the constant term, f0, in the objective function

.. index:: pair: variable; unitm
.. _doxid-structglrt__control__type_unitm:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool unitm

is M the identity matrix ?

.. index:: pair: variable; impose_descent
.. _doxid-structglrt__control__type_impose_descent:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool impose_descent

is descent required i.e., should $c^T x < 0$?

.. index:: pair: variable; space_critical
.. _doxid-structglrt__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structglrt__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; print_ritz_values
.. _doxid-structglrt__control__type_print_ritz_values:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool print_ritz_values

should the Ritz values be written to the debug stream?

.. index:: pair: variable; ritz_file_name
.. _doxid-structglrt__control__type_ritz_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char ritz_file_name[31]

name of debug file containing the Ritz values

.. index:: pair: variable; prefix
.. _doxid-structglrt__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

