.. index:: pair: struct; blls_control_type
.. _doxid-structblls__control__type:

blls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct blls_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          print_gap::INT
          maxit::INT
          cold_start::INT
          preconditioner::INT
          ratio_cg_vs_sd::INT
          change_max::INT
          cg_maxit::INT
          arcsearch_max_steps::INT
          sif_file_device::INT
          weight::T
          infinity::T
          stop_d::T
          identical_bounds_tol::T
          stop_cg_relative::T
          stop_cg_absolute::T
          alpha_max::T
          alpha_initial::T
          alpha_reduction::T
          arcsearch_acceptance_tol::T
          stabilisation_weight::T
          cpu_time_limit::T
          direct_subproblem_solve::Bool
          exact_arc_search::Bool
          advance::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          generate_sif_file::Bool
          sif_file_name::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sbls_control::sbls_control_type{T,INT}
          convert_control::convert_control_type{INT}

.. _details-structblls__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structblls__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structblls__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit number for error and warning diagnostics

.. index:: pair: variable; out
.. _doxid-structblls__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output unit number

.. index:: pair: variable; print_level
.. _doxid-structblls__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required

.. index:: pair: variable; start_print
.. _doxid-structblls__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

on which iteration to start printing

.. index:: pair: variable; stop_print
.. _doxid-structblls__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

on which iteration to stop printing

.. index:: pair: variable; print_gap
.. _doxid-structblls__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

how many iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structblls__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

how many iterations to perform (-ve reverts to HUGE(1)-1)

.. index:: pair: variable; cold_start
.. _doxid-structblls__control__type_cold_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cold_start

cold_start should be set to 0 if a warm start is required (with variable assigned according to X_stat, see below), and to any other value if the values given in prob.X suffice

.. index:: pair: variable; preconditioner
.. _doxid-structblls__control__type_preconditioner:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT preconditioner

the preconditioner (scaling) used. Possible values are: /li 0. no preconditioner. /li 1. a diagonal preconditioner that normalizes the rows of $A$. /li anything else. a preconditioner supplied by the user either via a subroutine call of eval_prec} or via reverse communication.

.. index:: pair: variable; ratio_cg_vs_sd
.. _doxid-structblls__control__type_ratio_cg_vs_sd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ratio_cg_vs_sd

the ratio of how many iterations use CGLS rather than steepest descent

.. index:: pair: variable; change_max
.. _doxid-structblls__control__type_change_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT change_max

the maximum number of per-iteration changes in the working set permitted when allowing CGLS rather than steepest descent

.. index:: pair: variable; cg_maxit
.. _doxid-structblls__control__type_cg_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_maxit

how many CG iterations to perform per BLLS iteration (-ve reverts to n+1)

.. index:: pair: variable; arcsearch_max_steps
.. _doxid-structblls__control__type_arcsearch_max_steps:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT arcsearch_max_steps

the maximum number of steps allowed in a piecewise arcsearch (-ve=infini

.. index:: pair: variable; sif_file_device
.. _doxid-structblls__control__type_sif_file_device:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sif_file_device

the unit number to write generated SIF file describing the current probl

.. index:: pair: variable; weight
.. _doxid-structblls__control__type_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight

the value of the non-negative regularization weight sigma, i.e., the quadratic objective function q(x) will be regularized by adding 1/2 weight \|\|x\|\|^2; any value smaller than zero will be regarded as zero.

.. index:: pair: variable; infinity
.. _doxid-structblls__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_d
.. _doxid-structblls__control__type_stop_d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structblls__control__type_identical_bounds_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T identical_bounds_tol

any pair of constraint bounds (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; stop_cg_relative
.. _doxid-structblls__control__type_stop_cg_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute)

.. index:: pair: variable; alpha_max
.. _doxid-structblls__control__type_alpha_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T alpha_max

the largest permitted arc length during the piecewise line search

.. index:: pair: variable; alpha_initial
.. _doxid-structblls__control__type_alpha_initial:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T alpha_initial

the initial arc length during the inexact piecewise line search

.. index:: pair: variable; alpha_reduction
.. _doxid-structblls__control__type_alpha_reduction:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T alpha_reduction

the arc length reduction factor for the inexact piecewise line search

.. index:: pair: variable; arcsearch_acceptance_tol
.. _doxid-structblls__control__type_arcsearch_acceptance_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T arcsearch_acceptance_tol

the required relative reduction during the inexact piecewise line search

.. index:: pair: variable; stabilisation_weight
.. _doxid-structblls__control__type_stabilisation_weight:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stabilisation_weight

the stabilisation weight added to the search-direction subproblem

.. index:: pair: variable; cpu_time_limit
.. _doxid-structblls__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve = no limit)

.. index:: pair: variable; direct_subproblem_solve
.. _doxid-structblls__control__type_direct_subproblem_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool direct_subproblem_solve

direct_subproblem_solve is true if the least-squares subproblem is to be solved using a matrix factorization, and false if conjugate gradients are to be preferred

.. index:: pair: variable; exact_arc_search
.. _doxid-structblls__control__type_exact_arc_search:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool exact_arc_search

exact_arc_search is true if an exact arc_search is required, and false if an approximation suffices

.. index:: pair: variable; advance
.. _doxid-structblls__control__type_advance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool advance

advance is true if an inexact exact arc_search can increase steps as well as decrease them

.. index:: pair: variable; space_critical
.. _doxid-structblls__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation times

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structblls__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structblls__control__type_generate_sif_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_sif_file

if generate_sif_file is true, a SIF file describing the current problem will be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structblls__control__type_sif_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} sif_file_name

name (max 30 characters) of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structblls__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by a string (max 30 characters) prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structblls__control__type_sbls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; convert_control
.. _doxid-structblls__control__type_convert_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`convert_control_type<doxid-structconvert__control__type>` convert_control

control parameters for CONVERT

