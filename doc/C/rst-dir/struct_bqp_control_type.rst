.. index:: pair: struct; bqp_control_type
.. _doxid-structbqp__control__type:

bqp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bqp.h>
	
	struct bqp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structbqp__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structbqp__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structbqp__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structbqp__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structbqp__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structbqp__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structbqp__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structbqp__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cold_start<doxid-structbqp__control__type_cold_start>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ratio_cg_vs_sd<doxid-structbqp__control__type_ratio_cg_vs_sd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`change_max<doxid-structbqp__control__type_change_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structbqp__control__type_cg_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structbqp__control__type_sif_file_device>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structbqp__control__type_infinity>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_p<doxid-structbqp__control__type_stop_p>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_d<doxid-structbqp__control__type_stop_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c<doxid-structbqp__control__type_stop_c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structbqp__control__type_identical_bounds_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_relative<doxid-structbqp__control__type_stop_cg_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_absolute<doxid-structbqp__control__type_stop_cg_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_curvature<doxid-structbqp__control__type_zero_curvature>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structbqp__control__type_cpu_time_limit>`;
		bool :ref:`exact_arcsearch<doxid-structbqp__control__type_exact_arcsearch>`;
		bool :ref:`space_critical<doxid-structbqp__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structbqp__control__type_deallocate_error_fatal>`;
		bool :ref:`generate_sif_file<doxid-structbqp__control__type_generate_sif_file>`;
		char :ref:`sif_file_name<doxid-structbqp__control__type_sif_file_name>`[31];
		char :ref:`prefix<doxid-structbqp__control__type_prefix>`[31];
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structbqp__control__type_sbls_control>`;
	};
.. _details-structbqp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbqp__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbqp__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit number for error and warning diagnostics

.. index:: pair: variable; out
.. _doxid-structbqp__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output unit number

.. index:: pair: variable; print_level
.. _doxid-structbqp__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required

.. index:: pair: variable; start_print
.. _doxid-structbqp__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

on which iteration to start printing

.. index:: pair: variable; stop_print
.. _doxid-structbqp__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

on which iteration to stop printing

.. index:: pair: variable; print_gap
.. _doxid-structbqp__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

how many iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structbqp__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

how many iterations to perform (-ve reverts to HUGE(1)-1)

.. index:: pair: variable; cold_start
.. _doxid-structbqp__control__type_cold_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cold_start

cold_start should be set to 0 if a warm start is required (with variable assigned according to B_stat, see below), and to any other value if the values given in prob.X suffice

.. index:: pair: variable; ratio_cg_vs_sd
.. _doxid-structbqp__control__type_ratio_cg_vs_sd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ratio_cg_vs_sd

the ratio of how many iterations use CG rather steepest descent

.. index:: pair: variable; change_max
.. _doxid-structbqp__control__type_change_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` change_max

the maximum number of per-iteration changes in the working set permitted when allowing CG rather than steepest descent

.. index:: pair: variable; cg_maxit
.. _doxid-structbqp__control__type_cg_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

how many CG iterations to perform per BQP iteration (-ve reverts to n+1)

.. index:: pair: variable; sif_file_device
.. _doxid-structbqp__control__type_sif_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structbqp__control__type_infinity:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_p
.. _doxid-structbqp__control__type_stop_p:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_p

the required accuracy for the primal infeasibility

.. index:: pair: variable; stop_d
.. _doxid-structbqp__control__type_stop_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_c
.. _doxid-structbqp__control__type_stop_c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c

the required accuracy for the complementary slackness

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structbqp__control__type_identical_bounds_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (x_l,x_u) that are closer than i dentical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; stop_cg_relative
.. _doxid-structbqp__control__type_stop_cg_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute)

.. index:: pair: variable; stop_cg_absolute
.. _doxid-structbqp__control__type_stop_cg_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_absolute

see stop_cg_relative

.. index:: pair: variable; zero_curvature
.. _doxid-structbqp__control__type_zero_curvature:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_curvature

threshold below which curvature is regarded as zero

.. index:: pair: variable; cpu_time_limit
.. _doxid-structbqp__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve = no limit)

.. index:: pair: variable; exact_arcsearch
.. _doxid-structbqp__control__type_exact_arcsearch:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_arcsearch

exact_arcsearch is true if an exact arcsearch is required, and false if approximation suffices

.. index:: pair: variable; space_critical
.. _doxid-structbqp__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation times

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbqp__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structbqp__control__type_generate_sif_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if generate_sif_file is true, a SIF file describing the current problem will be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structbqp__control__type_sif_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name (max 30 characters) of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structbqp__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by a string (max 30 characters) prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structbqp__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

