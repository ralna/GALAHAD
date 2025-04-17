.. index:: pair: struct; lpa_control_type
.. _doxid-structlpa__control__type:

lpa_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lpa_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          maxit::INT
          max_iterative_refinements::INT
          min_real_factor_size::INT
          min_integer_factor_size::INT
          random_number_seed::INT
          sif_file_device::INT
          qplib_file_device::INT
          infinity::T
          tol_data::T
          feas_tol::T
          relative_pivot_tolerance::T
          growth_limit::T
          zero_tolerance::T
          change_tolerance::T
          identical_bounds_tol::T
          cpu_time_limit::T
          clock_time_limit::T
          scale::Bool
          dual::Bool
          warm_start::Bool
          steepest_edge::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          generate_sif_file::Bool
          generate_qplib_file::Bool
          sif_file_name::NTuple{31,Cchar}
          qplib_file_name::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
	
.. _details-structlpa__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlpa__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlpa__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlpa__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlpa__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level (>= 2 turns on LA04 output)

.. index:: pair: variable; start_print
.. _doxid-structlpa__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlpa__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structlpa__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; max_iterative_refinements
.. _doxid-structlpa__control__type_max_iterative_refinements:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_iterative_refinements

maximum number of iterative refinements allowed

.. index:: pair: variable; min_real_factor_size
.. _doxid-structlpa__control__type_min_real_factor_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT min_real_factor_size

initial size for real array for the factors and other data

.. index:: pair: variable; min_integer_factor_size
.. _doxid-structlpa__control__type_min_integer_factor_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT min_integer_factor_size

initial size for integer array for the factors and other data

.. index:: pair: variable; random_number_seed
.. _doxid-structlpa__control__type_random_number_seed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT random_number_seed

the initial seed used when generating random numbers

.. index:: pair: variable; sif_file_device
.. _doxid-structlpa__control__type_sif_file_device:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; qplib_file_device
.. _doxid-structlpa__control__type_qplib_file_device:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT qplib_file_device

specifies the unit number to write generated QPLIB file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structlpa__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; tol_data
.. _doxid-structlpa__control__type_tol_data:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T tol_data

the tolerable relative perturbation of the data (A,g,..) defining the problem

.. index:: pair: variable; feas_tol
.. _doxid-structlpa__control__type_feas_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T feas_tol

any constraint violated by less than feas_tol will be considered to be satisfied

.. index:: pair: variable; relative_pivot_tolerance
.. _doxid-structlpa__control__type_relative_pivot_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T relative_pivot_tolerance

pivot threshold used to control the selection of pivot elements in the matrix factorization. Any potential pivot which is less than the largest entry in its row times the threshold is excluded as a candidate

.. index:: pair: variable; growth_limit
.. _doxid-structlpa__control__type_growth_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T growth_limit

limit to control growth in the upated basis factors. A refactorization occurs if the growth exceeds this limit

.. index:: pair: variable; zero_tolerance
.. _doxid-structlpa__control__type_zero_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_tolerance

any entry in the basis smaller than this is considered zero

.. index:: pair: variable; change_tolerance
.. _doxid-structlpa__control__type_change_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T change_tolerance

any solution component whose change is smaller than a tolerence times the largest change may be considered to be zero

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structlpa__control__type_identical_bounds_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; cpu_time_limit
.. _doxid-structlpa__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structlpa__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; scale
.. _doxid-structlpa__control__type_scale:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool scale

if .scale is true, the problem will be automatically scaled prior to solution. This may improve computation time and accuracy

.. index:: pair: variable; dual
.. _doxid-structlpa__control__type_dual:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool dual

should the dual problem be solved rather than the primal?

.. index:: pair: variable; warm_start
.. _doxid-structlpa__control__type_warm_start:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool warm_start

should a warm start using the data in C_stat and X_stat be attempted?

.. index:: pair: variable; steepest_edge
.. _doxid-structlpa__control__type_steepest_edge:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool steepest_edge

should steepest-edge weights be used to detetrmine the variable leaving the basis?

.. index:: pair: variable; space_critical
.. _doxid-structlpa__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlpa__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structlpa__control__type_generate_sif_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; generate_qplib_file
.. _doxid-structlpa__control__type_generate_qplib_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_qplib_file

if .generate_qplib_file is .true. if a QPLIB file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structlpa__control__type_sif_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} sif_file_name

name of generated SIF file containing input problem

.. index:: pair: variable; qplib_file_name
.. _doxid-structlpa__control__type_qplib_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} qplib_file_name

name of generated QPLIB file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structlpa__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

