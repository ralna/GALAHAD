.. index:: pair: struct; slls_control_type
.. _doxid-structslls__control__type:

slls_control_type structure
---------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_slls.h>
	
	struct slls_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structslls__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structslls__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structslls__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structslls__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structslls__control__type_start_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structslls__control__type_stop_print>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structslls__control__type_print_gap>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structslls__control__type_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cold_start<doxid-structslls__control__type_cold_start>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structslls__control__type_preconditioner>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ratio_cg_vs_sd<doxid-structslls__control__type_ratio_cg_vs_sd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`change_max<doxid-structslls__control__type_change_max>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structslls__control__type_cg_maxit>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`arcsearch_max_steps<doxid-structslls__control__type_arcsearch_max_steps>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structslls__control__type_sif_file_device>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structslls__control__type_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_d<doxid-structslls__control__type_stop_d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_relative<doxid-structslls__control__type_stop_cg_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :target:`stop_cg_absolute<doxid-structslls__control__type_stop_cg_absolute>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_max<doxid-structslls__control__type_alpha_max>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_initial<doxid-structslls__control__type_alpha_initial>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_reduction<doxid-structslls__control__type_alpha_reduction>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`arcsearch_acceptance_tol<doxid-structslls__control__type_arcsearch_acceptance_tol>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stabilisation_weight<doxid-structslls__control__type_stabilisation_weight>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structslls__control__type_cpu_time_limit>`;
		bool :ref:`direct_subproblem_solve<doxid-structslls__control__type_direct_subproblem_solve>`;
		bool :ref:`exact_arc_search<doxid-structslls__control__type_exact_arc_search>`;
		bool :ref:`space_critical<doxid-structslls__control__type_space_critical>`;
		bool :ref:`deallocate_error_fatal<doxid-structslls__control__type_deallocate_error_fatal>`;
		bool :ref:`generate_sif_file<doxid-structslls__control__type_generate_sif_file>`;
		char :ref:`sif_file_name<doxid-structslls__control__type_sif_file_name>`[31];
		char :ref:`prefix<doxid-structslls__control__type_prefix>`[31];
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structslls__control__type_sbls_control>`;
		struct :ref:`convert_control_type<doxid-structconvert__control__type>` :ref:`convert_control<doxid-structslls__control__type_convert_control>`;
	};
.. _details-structslls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structslls__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structslls__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit number for error and warning diagnostics

.. index:: pair: variable; out
.. _doxid-structslls__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output unit number

.. index:: pair: variable; print_level
.. _doxid-structslls__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required

.. index:: pair: variable; start_print
.. _doxid-structslls__control__type_start_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

on which iteration to start printing

.. index:: pair: variable; stop_print
.. _doxid-structslls__control__type_stop_print:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

on which iteration to stop printing

.. index:: pair: variable; print_gap
.. _doxid-structslls__control__type_print_gap:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

how many iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structslls__control__type_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

how many iterations to perform (-ve reverts to HUGE(1)-1)

.. index:: pair: variable; cold_start
.. _doxid-structslls__control__type_cold_start:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cold_start

cold_start should be set to 0 if a warm start is required (with variable assigned according to X_stat, see below), and to any other value if the values given in prob.X suffice

.. index:: pair: variable; preconditioner
.. _doxid-structslls__control__type_preconditioner:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

the preconditioner (scaling) used. Possible values are: /li 0. no preconditioner. /li 1. a diagonal preconditioner that normalizes the rows of $A$. /li anything else. a preconditioner supplied by the user either via a subroutine call of eval_prec} or via reverse communication.

.. index:: pair: variable; ratio_cg_vs_sd
.. _doxid-structslls__control__type_ratio_cg_vs_sd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ratio_cg_vs_sd

the ratio of how many iterations use CGLS rather than steepest descent

.. index:: pair: variable; change_max
.. _doxid-structslls__control__type_change_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` change_max

the maximum number of per-iteration changes in the working set permitted when allowing CGLS rather than steepest descent

.. index:: pair: variable; cg_maxit
.. _doxid-structslls__control__type_cg_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

how many CG iterations to perform per SLLS iteration (-ve reverts to n+1)

.. index:: pair: variable; arcsearch_max_steps
.. _doxid-structslls__control__type_arcsearch_max_steps:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` arcsearch_max_steps

the maximum number of steps allowed in a piecewise arcsearch (-ve=infini

.. index:: pair: variable; sif_file_device
.. _doxid-structslls__control__type_sif_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

the unit number to write generated SIF file describing the current probl

.. index:: pair: variable; weight
.. _doxid-structslls__control__type_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the value of the non-negative regularization weight sigma, i.e., the quadratic objective function q(x) will be regularized by adding 1/2 weight \|\|x\|\|^2; any value smaller than zero will be regarded as zero.

.. index:: pair: variable; stop_d
.. _doxid-structslls__control__type_stop_d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_cg_relative
.. _doxid-structslls__control__type_stop_cg_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute)

.. index:: pair: variable; alpha_max
.. _doxid-structslls__control__type_alpha_max:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_max

the largest permitted arc length during the piecewise line search

.. index:: pair: variable; alpha_initial
.. _doxid-structslls__control__type_alpha_initial:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_initial

the initial arc length during the inexact piecewise line search

.. index:: pair: variable; alpha_reduction
.. _doxid-structslls__control__type_alpha_reduction:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_reduction

the arc length reduction factor for the inexact piecewise line search

.. index:: pair: variable; arcsearch_acceptance_tol
.. _doxid-structslls__control__type_arcsearch_acceptance_tol:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` arcsearch_acceptance_tol

the required relative reduction during the inexact piecewise line search

.. index:: pair: variable; stabilisation_weight
.. _doxid-structslls__control__type_stabilisation_weight:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stabilisation_weight

the stabilisation weight added to the search-direction subproblem

.. index:: pair: variable; cpu_time_limit
.. _doxid-structslls__control__type_cpu_time_limit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve = no limit)

.. index:: pair: variable; direct_subproblem_solve
.. _doxid-structslls__control__type_direct_subproblem_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool direct_subproblem_solve

direct_subproblem_solve is true if the least-squares subproblem is to be solved using a matrix factorization, and false if conjugate gradients are to be preferred

.. index:: pair: variable; exact_arc_search
.. _doxid-structslls__control__type_exact_arc_search:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_arc_search

exact_arc_search is true if an exact arc_search is required, and false if an approximation suffices

.. index:: pair: variable; space_critical
.. _doxid-structslls__control__type_space_critical:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation times

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structslls__control__type_deallocate_error_fatal:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structslls__control__type_generate_sif_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if generate_sif_file is true, a SIF file describing the current problem will be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structslls__control__type_sif_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name (max 30 characters) of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structslls__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by a string (max 30 characters) prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structslls__control__type_sbls_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; convert_control
.. _doxid-structslls__control__type_convert_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`convert_control_type<doxid-structconvert__control__type>` convert_control

control parameters for CONVERT

