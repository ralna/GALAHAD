.. index:: pair: table; sls_control_type
.. _doxid-structsls__control__type:

sls_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sls.h>
	
	struct sls_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structsls__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structsls__control__type_error>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`warning<doxid-structsls__control__type_warning>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structsls__control__type_out>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`statistics<doxid-structsls__control__type_statistics>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structsls__control__type_print_level>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level_solver<doxid-structsls__control__type_print_level_solver>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`bits<doxid-structsls__control__type_bits>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`block_size_kernel<doxid-structsls__control__type_block_size_kernel>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`block_size_elimination<doxid-structsls__control__type_block_size_elimination>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`blas_block_size_factorize<doxid-structsls__control__type_blas_block_size_factorize>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`blas_block_size_solve<doxid-structsls__control__type_blas_block_size_solve>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`node_amalgamation<doxid-structsls__control__type_node_amalgamation>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`initial_pool_size<doxid-structsls__control__type_initial_pool_size>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`min_real_factor_size<doxid-structsls__control__type_min_real_factor_size>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`min_integer_factor_size<doxid-structsls__control__type_min_integer_factor_size>`;
		int64_t :ref:`max_real_factor_size<doxid-structsls__control__type_max_real_factor_size>`;
		int64_t :ref:`max_integer_factor_size<doxid-structsls__control__type_max_integer_factor_size>`;
		int64_t :ref:`max_in_core_store<doxid-structsls__control__type_max_in_core_store>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`array_increase_factor<doxid-structsls__control__type_array_increase_factor>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`array_decrease_factor<doxid-structsls__control__type_array_decrease_factor>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pivot_control<doxid-structsls__control__type_pivot_control>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ordering<doxid-structsls__control__type_ordering>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`full_row_threshold<doxid-structsls__control__type_full_row_threshold>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`row_search_indefinite<doxid-structsls__control__type_row_search_indefinite>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scaling<doxid-structsls__control__type_scaling>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scale_maxit<doxid-structsls__control__type_scale_maxit>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`scale_thresh<doxid-structsls__control__type_scale_thresh>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`relative_pivot_tolerance<doxid-structsls__control__type_relative_pivot_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_pivot_tolerance<doxid-structsls__control__type_minimum_pivot_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`absolute_pivot_tolerance<doxid-structsls__control__type_absolute_pivot_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_tolerance<doxid-structsls__control__type_zero_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot_tolerance<doxid-structsls__control__type_zero_pivot_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`negative_pivot_tolerance<doxid-structsls__control__type_negative_pivot_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_pivot_tolerance<doxid-structsls__control__type_static_pivot_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_level_switch<doxid-structsls__control__type_static_level_switch>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`consistency_tolerance<doxid-structsls__control__type_consistency_tolerance>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_iterative_refinements<doxid-structsls__control__type_max_iterative_refinements>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_relative<doxid-structsls__control__type_acceptable_residual_relative>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_absolute<doxid-structsls__control__type_acceptable_residual_absolute>`;
		bool :ref:`multiple_rhs<doxid-structsls__control__type_multiple_rhs>`;
		bool :ref:`generate_matrix_file<doxid-structsls__control__type_generate_matrix_file>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`matrix_file_device<doxid-structsls__control__type_matrix_file_device>`;
		char :ref:`matrix_file_name<doxid-structsls__control__type_matrix_file_name>`[31];
		char :ref:`out_of_core_directory<doxid-structsls__control__type_out_of_core_directory>`[401];
		char :ref:`out_of_core_integer_factor_file<doxid-structsls__control__type_out_of_core_integer_factor_file>`[401];
		char :ref:`out_of_core_real_factor_file<doxid-structsls__control__type_out_of_core_real_factor_file>`[401];
		char :ref:`out_of_core_real_work_file<doxid-structsls__control__type_out_of_core_real_work_file>`[401];
		char :ref:`out_of_core_indefinite_file<doxid-structsls__control__type_out_of_core_indefinite_file>`[401];
		char :ref:`out_of_core_restart_file<doxid-structsls__control__type_out_of_core_restart_file>`[501];
		char :ref:`prefix<doxid-structsls__control__type_prefix>`[31];
		struct nodend_control_type :ref:`nodend_control<doxid-structsls__control__type_nodend_control>`;
	};
.. _details-structsls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsls__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsls__control__type_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; warning
.. _doxid-structsls__control__type_warning:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` warning

unit for warning messages

.. index:: pair: variable; out
.. _doxid-structsls__control__type_out:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; statistics
.. _doxid-structsls__control__type_statistics:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` statistics

unit for statistical output

.. index:: pair: variable; print_level
.. _doxid-structsls__control__type_print_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; print_level_solver
.. _doxid-structsls__control__type_print_level_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level_solver

controls level of diagnostic output from external solver

.. index:: pair: variable; bits
.. _doxid-structsls__control__type_bits:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` bits

number of bits used in architecture

.. index:: pair: variable; block_size_kernel
.. _doxid-structsls__control__type_block_size_kernel:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` block_size_kernel

the target blocksize for kernel factorization

.. index:: pair: variable; block_size_elimination
.. _doxid-structsls__control__type_block_size_elimination:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` block_size_elimination

the target blocksize for parallel elimination

.. index:: pair: variable; blas_block_size_factorize
.. _doxid-structsls__control__type_blas_block_size_factorize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` blas_block_size_factorize

level 3 blocking in factorize

.. index:: pair: variable; blas_block_size_solve
.. _doxid-structsls__control__type_blas_block_size_solve:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` blas_block_size_solve

level 2 and 3 blocking in solve

.. index:: pair: variable; node_amalgamation
.. _doxid-structsls__control__type_node_amalgamation:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` node_amalgamation

a child node is merged with its parent if they both involve fewer than node_amalgamation eliminations

.. index:: pair: variable; initial_pool_size
.. _doxid-structsls__control__type_initial_pool_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` initial_pool_size

initial size of task-pool arrays for parallel elimination

.. index:: pair: variable; min_real_factor_size
.. _doxid-structsls__control__type_min_real_factor_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` min_real_factor_size

initial size for real array for the factors and other data

.. index:: pair: variable; min_integer_factor_size
.. _doxid-structsls__control__type_min_integer_factor_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` min_integer_factor_size

initial size for integer array for the factors and other data

.. index:: pair: variable; max_real_factor_size
.. _doxid-structsls__control__type_max_real_factor_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_real_factor_size

maximum size for real array for the factors and other data

.. index:: pair: variable; max_integer_factor_size
.. _doxid-structsls__control__type_max_integer_factor_size:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_integer_factor_size

maximum size for integer array for the factors and other data

.. index:: pair: variable; max_in_core_store
.. _doxid-structsls__control__type_max_in_core_store:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_in_core_store

amount of in-core storage to be used for out-of-core factorization

.. index:: pair: variable; array_increase_factor
.. _doxid-structsls__control__type_array_increase_factor:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` array_increase_factor

factor by which arrays sizes are to be increased if they are too small

.. index:: pair: variable; array_decrease_factor
.. _doxid-structsls__control__type_array_decrease_factor:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` array_decrease_factor

if previously allocated internal workspace arrays are greater than array_decrease_factor times the currently required sizes, they are reset to current requirements

.. index:: pair: variable; pivot_control
.. _doxid-structsls__control__type_pivot_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pivot_control

pivot control:

* 1 Numerical pivoting will be performed.

* 2 No pivoting will be performed and an error exit will occur immediately a pivot sign change is detected.

* 3 No pivoting will be performed and an error exit will occur if a zero pivot is detected.

* 4 No pivoting is performed but pivots are changed to all be positive

.. index:: pair: variable; ordering
.. _doxid-structsls__control__type_ordering:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ordering

controls ordering (ignored if explicit PERM argument present)

* <0 chosen by the specified solver with its own ordering-selected value -ordering

* 0 chosen package default (or the AMD ordering if no package default)

* 1 Approximate minimum degree (AMD) with provisions for "dense" rows/col

* 2 Minimum degree

* 3 Nested disection

* 4 indefinite ordering to generate a combination of 1x1 and 2x2 pivots

* 5 Profile/Wavefront reduction

* 6 Bandwidth reduction

* >6 ordering chosen depending on matrix characteristics (not yet implemented)

.. index:: pair: variable; full_row_threshold
.. _doxid-structsls__control__type_full_row_threshold:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` full_row_threshold

controls threshold for detecting full rows in analyse, registered as percentage of matrix order. If 100, only fully dense rows detected (defa

.. index:: pair: variable; row_search_indefinite
.. _doxid-structsls__control__type_row_search_indefinite:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` row_search_indefinite

number of rows searched for pivot when using indefinite ordering

.. index:: pair: variable; scaling
.. _doxid-structsls__control__type_scaling:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scaling

controls scaling (ignored if explicit SCALE argument present)

* <0 chosen by the specified solver with its own scaling-selected value -scaling

* 0 No scaling

* 1 Scaling using HSL's MC64

* 2 Scaling using HSL's MC77 based on the row one-norm

* 3 Scaling using HSL's MC77 based on the row infinity-norm

.. index:: pair: variable; scale_maxit
.. _doxid-structsls__control__type_scale_maxit:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scale_maxit

the number of scaling iterations performed (default 10 used if .scale_maxit < 0)

.. index:: pair: variable; scale_thresh
.. _doxid-structsls__control__type_scale_thresh:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` scale_thresh

the scaling iteration stops as soon as the row/column norms are less than 1+/-.scale_thresh

.. index:: pair: variable; relative_pivot_tolerance
.. _doxid-structsls__control__type_relative_pivot_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` relative_pivot_tolerance

pivot threshold

.. index:: pair: variable; minimum_pivot_tolerance
.. _doxid-structsls__control__type_minimum_pivot_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_pivot_tolerance

smallest permitted relative pivot threshold

.. index:: pair: variable; absolute_pivot_tolerance
.. _doxid-structsls__control__type_absolute_pivot_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` absolute_pivot_tolerance

any pivot small than this is considered zero

.. index:: pair: variable; zero_tolerance
.. _doxid-structsls__control__type_zero_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_tolerance

any entry smaller than this is considered zero

.. index:: pair: variable; zero_pivot_tolerance
.. _doxid-structsls__control__type_zero_pivot_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot_tolerance

any pivot smaller than this is considered zero for positive-definite sol

.. index:: pair: variable; negative_pivot_tolerance
.. _doxid-structsls__control__type_negative_pivot_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` negative_pivot_tolerance

any pivot smaller than this is considered to be negative for p-d solvers

.. index:: pair: variable; static_pivot_tolerance
.. _doxid-structsls__control__type_static_pivot_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_pivot_tolerance

used for setting static pivot level

.. index:: pair: variable; static_level_switch
.. _doxid-structsls__control__type_static_level_switch:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_level_switch

used for switch to static

.. index:: pair: variable; consistency_tolerance
.. _doxid-structsls__control__type_consistency_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` consistency_tolerance

used to determine whether a system is consistent when seeking a Fredholm alternative

.. index:: pair: variable; max_iterative_refinements
.. _doxid-structsls__control__type_max_iterative_refinements:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_iterative_refinements

maximum number of iterative refinements allowed

.. index:: pair: variable; acceptable_residual_relative
.. _doxid-structsls__control__type_acceptable_residual_relative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_relative

refinement will cease as soon as the residual \|\|Ax-b\|\| falls below max( acceptable_residual_relative \* \|\|b\|\|, acceptable_residual_absolute

.. index:: pair: variable; acceptable_residual_absolute
.. _doxid-structsls__control__type_acceptable_residual_absolute:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_absolute

see acceptable_residual_relative

.. index:: pair: variable; multiple_rhs
.. _doxid-structsls__control__type_multiple_rhs:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool multiple_rhs

set .multiple_rhs to .true. if there is possibility that the solver will be required to solve systems with more than one right-hand side. More efficient execution may be possible when .multiple_rhs = .false.

.. index:: pair: variable; generate_matrix_file
.. _doxid-structsls__control__type_generate_matrix_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_matrix_file

if .generate_matrix_file is .true. if a file describing the current matrix is to be generated

.. index:: pair: variable; matrix_file_device
.. _doxid-structsls__control__type_matrix_file_device:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` matrix_file_device

specifies the unit number to write the input matrix (in co-ordinate form

.. index:: pair: variable; matrix_file_name
.. _doxid-structsls__control__type_matrix_file_name:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char matrix_file_name[31]

name of generated matrix file containing input problem

.. index:: pair: variable; out_of_core_directory
.. _doxid-structsls__control__type_out_of_core_directory:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_directory[401]

directory name for out of core factorization and additional real workspace in the indefinite case, respectively

.. index:: pair: variable; out_of_core_integer_factor_file
.. _doxid-structsls__control__type_out_of_core_integer_factor_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_integer_factor_file[401]

out of core superfile names for integer and real factor data, real works and additional real workspace in the indefinite case, respectively

.. index:: pair: variable; out_of_core_real_factor_file
.. _doxid-structsls__control__type_out_of_core_real_factor_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_real_factor_file[401]

see out_of_core_integer_factor_file

.. index:: pair: variable; out_of_core_real_work_file
.. _doxid-structsls__control__type_out_of_core_real_work_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_real_work_file[401]

see out_of_core_integer_factor_file

.. index:: pair: variable; out_of_core_indefinite_file
.. _doxid-structsls__control__type_out_of_core_indefinite_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_indefinite_file[401]

see out_of_core_integer_factor_file

.. index:: pair: variable; out_of_core_restart_file
.. _doxid-structsls__control__type_out_of_core_restart_file:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_restart_file[501]

see out_of_core_integer_factor_file

.. index:: pair: variable; prefix
.. _doxid-structsls__control__type_prefix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; nodend_control
.. _doxid-structsls__control__type_nodend_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`nodend_control_type<doxid-structnodend__control__type>` nodend_control

the input control structure for nodend
