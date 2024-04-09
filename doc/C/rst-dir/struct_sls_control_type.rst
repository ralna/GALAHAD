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
	
		bool :ref:`f_indexing<doxid-structsls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structsls__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`warning<doxid-structsls__control__type_1af44472f3bdd4c3af6ed5465266b25312>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structsls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`statistics<doxid-structsls__control__type_1ac1dde3b0d96e2c1bc0c226bc3da59846>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structsls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level_solver<doxid-structsls__control__type_1afc17ef86601030a3cb1c7edb7a79a39f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`bits<doxid-structsls__control__type_1ad1e28a1a66a25529b0b61b9ca4e66d44>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`block_size_kernel<doxid-structsls__control__type_1a6074142046e83fb79960ed41f26750a1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`block_size_elimination<doxid-structsls__control__type_1a8f2a7a17307a57f735f5d954d59e2eb8>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`blas_block_size_factorize<doxid-structsls__control__type_1aa2177c655d92533f17fcff58482e52c1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`blas_block_size_solve<doxid-structsls__control__type_1a1af3c20cdeaaee431f309395f9f09564>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`node_amalgamation<doxid-structsls__control__type_1ac3c6c4a0c684c675b397c1d07b4eb170>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`initial_pool_size<doxid-structsls__control__type_1ad5ba76433cbea285e63bf2bb54b4654c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`min_real_factor_size<doxid-structsls__control__type_1a60c3d0376bbe5c06260509a9be31c562>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`min_integer_factor_size<doxid-structsls__control__type_1addaddc306c33f5d7e6ebf7eeeab2612e>`;
		int64_t :ref:`max_real_factor_size<doxid-structsls__control__type_1aba2b38ee264ef80678efeff94ef0d44b>`;
		int64_t :ref:`max_integer_factor_size<doxid-structsls__control__type_1aa53de4fa276629aa7eef29736c72f980>`;
		int64_t :ref:`max_in_core_store<doxid-structsls__control__type_1a5eccee707d29802453ebc44a238211d9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`array_increase_factor<doxid-structsls__control__type_1a6189cbfe4360772c9fa6b554e93a9b2b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`array_decrease_factor<doxid-structsls__control__type_1abee09e4efae03be7d0d1a8503b338ce7>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pivot_control<doxid-structsls__control__type_1a7489428a42ce1420b4891f638153c99f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ordering<doxid-structsls__control__type_1a4175ebe476addcfc3433fc97c19e0708>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`full_row_threshold<doxid-structsls__control__type_1a3b64788f24cdaf2e3675ec0d030dc1b2>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`row_search_indefinite<doxid-structsls__control__type_1aae15c41cfdc178c1daa93e5d6cdee74a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scaling<doxid-structsls__control__type_1a26f0572eeeaa419eabb09dc89c00b89d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scale_maxit<doxid-structsls__control__type_1af5fd3fa336214d064c0573c95000440e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`scale_thresh<doxid-structsls__control__type_1a6e985cab4e94c7266dd26ed49264814b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`relative_pivot_tolerance<doxid-structsls__control__type_1a65344e4192516e9b621cc7416f09045c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`minimum_pivot_tolerance<doxid-structsls__control__type_1ab01168ba8df4956450d31f529999b8c6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`absolute_pivot_tolerance<doxid-structsls__control__type_1aa5e25bdcf567fac1fb496199a7c06d5a>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_tolerance<doxid-structsls__control__type_1afb83ea8401da6498362914aa88ae823f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot_tolerance<doxid-structsls__control__type_1aedcd7d4cdfcc81885bdcbf2c7135db7b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`negative_pivot_tolerance<doxid-structsls__control__type_1a05f2424e16a89a52908bb9035300c45d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_pivot_tolerance<doxid-structsls__control__type_1aa0dbb782fd6585b488abb9e62954cc0b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_level_switch<doxid-structsls__control__type_1ab6b34ef1c7cd56ca2906c4dc0ba7ec9b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`consistency_tolerance<doxid-structsls__control__type_1a7582103f0c0f9eac3b325bfff86236d1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_iterative_refinements<doxid-structsls__control__type_1ab044fd598767830ddc06560a91b80936>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_relative<doxid-structsls__control__type_1a97a6571829dbdccad7598f7b5c3ddfbd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`acceptable_residual_absolute<doxid-structsls__control__type_1a5ee0e70d90b1398019054b19b68057a0>`;
		bool :ref:`multiple_rhs<doxid-structsls__control__type_1a0e2ee73a2ee1899ac7aa07a56c9115d5>`;
		bool :ref:`generate_matrix_file<doxid-structsls__control__type_1ac6d360fda05100d7c8a5580a78e44c59>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`matrix_file_device<doxid-structsls__control__type_1a184e374e7f9b631289f30f2cbfd4f8a8>`;
		char :ref:`matrix_file_name<doxid-structsls__control__type_1a0861a0a49e00bd6642d2103f0d5fb7a3>`[31];
		char :ref:`out_of_core_directory<doxid-structsls__control__type_1af13c49b84e4bde8d45c258a969592a1d>`[401];
		char :ref:`out_of_core_integer_factor_file<doxid-structsls__control__type_1af99536361792bb2eb73942011fa1c5e9>`[401];
		char :ref:`out_of_core_real_factor_file<doxid-structsls__control__type_1a42bdeec279b8e8856ddc5fc64254b93d>`[401];
		char :ref:`out_of_core_real_work_file<doxid-structsls__control__type_1ac677c7831bafaddfc5bf60069fdc3171>`[401];
		char :ref:`out_of_core_indefinite_file<doxid-structsls__control__type_1a6e9acddbc56a33e87fb4532349da129b>`[401];
		char :ref:`out_of_core_restart_file<doxid-structsls__control__type_1a24a37f3b4cc3a36dfa2954f906a45e11>`[501];
		char :ref:`prefix<doxid-structsls__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structsls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structsls__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; warning
.. _doxid-structsls__control__type_1af44472f3bdd4c3af6ed5465266b25312:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` warning

unit for warning messages

.. index:: pair: variable; out
.. _doxid-structsls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; statistics
.. _doxid-structsls__control__type_1ac1dde3b0d96e2c1bc0c226bc3da59846:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` statistics

unit for statistical output

.. index:: pair: variable; print_level
.. _doxid-structsls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; print_level_solver
.. _doxid-structsls__control__type_1afc17ef86601030a3cb1c7edb7a79a39f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level_solver

controls level of diagnostic output from external solver

.. index:: pair: variable; bits
.. _doxid-structsls__control__type_1ad1e28a1a66a25529b0b61b9ca4e66d44:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` bits

number of bits used in architecture

.. index:: pair: variable; block_size_kernel
.. _doxid-structsls__control__type_1a6074142046e83fb79960ed41f26750a1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` block_size_kernel

the target blocksize for kernel factorization

.. index:: pair: variable; block_size_elimination
.. _doxid-structsls__control__type_1a8f2a7a17307a57f735f5d954d59e2eb8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` block_size_elimination

the target blocksize for parallel elimination

.. index:: pair: variable; blas_block_size_factorize
.. _doxid-structsls__control__type_1aa2177c655d92533f17fcff58482e52c1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` blas_block_size_factorize

level 3 blocking in factorize

.. index:: pair: variable; blas_block_size_solve
.. _doxid-structsls__control__type_1a1af3c20cdeaaee431f309395f9f09564:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` blas_block_size_solve

level 2 and 3 blocking in solve

.. index:: pair: variable; node_amalgamation
.. _doxid-structsls__control__type_1ac3c6c4a0c684c675b397c1d07b4eb170:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` node_amalgamation

a child node is merged with its parent if they both involve fewer than node_amalgamation eliminations

.. index:: pair: variable; initial_pool_size
.. _doxid-structsls__control__type_1ad5ba76433cbea285e63bf2bb54b4654c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` initial_pool_size

initial size of task-pool arrays for parallel elimination

.. index:: pair: variable; min_real_factor_size
.. _doxid-structsls__control__type_1a60c3d0376bbe5c06260509a9be31c562:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` min_real_factor_size

initial size for real array for the factors and other data

.. index:: pair: variable; min_integer_factor_size
.. _doxid-structsls__control__type_1addaddc306c33f5d7e6ebf7eeeab2612e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` min_integer_factor_size

initial size for integer array for the factors and other data

.. index:: pair: variable; max_real_factor_size
.. _doxid-structsls__control__type_1aba2b38ee264ef80678efeff94ef0d44b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_real_factor_size

maximum size for real array for the factors and other data

.. index:: pair: variable; max_integer_factor_size
.. _doxid-structsls__control__type_1aa53de4fa276629aa7eef29736c72f980:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_integer_factor_size

maximum size for integer array for the factors and other data

.. index:: pair: variable; max_in_core_store
.. _doxid-structsls__control__type_1a5eccee707d29802453ebc44a238211d9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t max_in_core_store

amount of in-core storage to be used for out-of-core factorization

.. index:: pair: variable; array_increase_factor
.. _doxid-structsls__control__type_1a6189cbfe4360772c9fa6b554e93a9b2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` array_increase_factor

factor by which arrays sizes are to be increased if they are too small

.. index:: pair: variable; array_decrease_factor
.. _doxid-structsls__control__type_1abee09e4efae03be7d0d1a8503b338ce7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` array_decrease_factor

if previously allocated internal workspace arrays are greater than array_decrease_factor times the currently required sizes, they are reset to current requirements

.. index:: pair: variable; pivot_control
.. _doxid-structsls__control__type_1a7489428a42ce1420b4891f638153c99f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pivot_control

pivot control:

* 1 Numerical pivoting will be performed.

* 2 No pivoting will be performed and an error exit will occur immediately a pivot sign change is detected.

* 3 No pivoting will be performed and an error exit will occur if a zero pivot is detected.

* 4 No pivoting is performed but pivots are changed to all be positive

.. index:: pair: variable; ordering
.. _doxid-structsls__control__type_1a4175ebe476addcfc3433fc97c19e0708:

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
.. _doxid-structsls__control__type_1a3b64788f24cdaf2e3675ec0d030dc1b2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` full_row_threshold

controls threshold for detecting full rows in analyse, registered as percentage of matrix order. If 100, only fully dense rows detected (defa

.. index:: pair: variable; row_search_indefinite
.. _doxid-structsls__control__type_1aae15c41cfdc178c1daa93e5d6cdee74a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` row_search_indefinite

number of rows searched for pivot when using indefinite ordering

.. index:: pair: variable; scaling
.. _doxid-structsls__control__type_1a26f0572eeeaa419eabb09dc89c00b89d:

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
.. _doxid-structsls__control__type_1af5fd3fa336214d064c0573c95000440e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scale_maxit

the number of scaling iterations performed (default 10 used if .scale_maxit < 0)

.. index:: pair: variable; scale_thresh
.. _doxid-structsls__control__type_1a6e985cab4e94c7266dd26ed49264814b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` scale_thresh

the scaling iteration stops as soon as the row/column norms are less than 1+/-.scale_thresh

.. index:: pair: variable; relative_pivot_tolerance
.. _doxid-structsls__control__type_1a65344e4192516e9b621cc7416f09045c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` relative_pivot_tolerance

pivot threshold

.. index:: pair: variable; minimum_pivot_tolerance
.. _doxid-structsls__control__type_1ab01168ba8df4956450d31f529999b8c6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` minimum_pivot_tolerance

smallest permitted relative pivot threshold

.. index:: pair: variable; absolute_pivot_tolerance
.. _doxid-structsls__control__type_1aa5e25bdcf567fac1fb496199a7c06d5a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` absolute_pivot_tolerance

any pivot small than this is considered zero

.. index:: pair: variable; zero_tolerance
.. _doxid-structsls__control__type_1afb83ea8401da6498362914aa88ae823f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_tolerance

any entry smaller than this is considered zero

.. index:: pair: variable; zero_pivot_tolerance
.. _doxid-structsls__control__type_1aedcd7d4cdfcc81885bdcbf2c7135db7b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot_tolerance

any pivot smaller than this is considered zero for positive-definite sol

.. index:: pair: variable; negative_pivot_tolerance
.. _doxid-structsls__control__type_1a05f2424e16a89a52908bb9035300c45d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` negative_pivot_tolerance

any pivot smaller than this is considered to be negative for p-d solvers

.. index:: pair: variable; static_pivot_tolerance
.. _doxid-structsls__control__type_1aa0dbb782fd6585b488abb9e62954cc0b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_pivot_tolerance

used for setting static pivot level

.. index:: pair: variable; static_level_switch
.. _doxid-structsls__control__type_1ab6b34ef1c7cd56ca2906c4dc0ba7ec9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_level_switch

used for switch to static

.. index:: pair: variable; consistency_tolerance
.. _doxid-structsls__control__type_1a7582103f0c0f9eac3b325bfff86236d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` consistency_tolerance

used to determine whether a system is consistent when seeking a Fredholm alternative

.. index:: pair: variable; max_iterative_refinements
.. _doxid-structsls__control__type_1ab044fd598767830ddc06560a91b80936:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_iterative_refinements

maximum number of iterative refinements allowed

.. index:: pair: variable; acceptable_residual_relative
.. _doxid-structsls__control__type_1a97a6571829dbdccad7598f7b5c3ddfbd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_relative

refinement will cease as soon as the residual \|\|Ax-b\|\| falls below max( acceptable_residual_relative \* \|\|b\|\|, acceptable_residual_absolute

.. index:: pair: variable; acceptable_residual_absolute
.. _doxid-structsls__control__type_1a5ee0e70d90b1398019054b19b68057a0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` acceptable_residual_absolute

see acceptable_residual_relative

.. index:: pair: variable; multiple_rhs
.. _doxid-structsls__control__type_1a0e2ee73a2ee1899ac7aa07a56c9115d5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool multiple_rhs

set .multiple_rhs to .true. if there is possibility that the solver will be required to solve systems with more than one right-hand side. More efficient execution may be possible when .multiple_rhs = .false.

.. index:: pair: variable; generate_matrix_file
.. _doxid-structsls__control__type_1ac6d360fda05100d7c8a5580a78e44c59:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_matrix_file

if .generate_matrix_file is .true. if a file describing the current matrix is to be generated

.. index:: pair: variable; matrix_file_device
.. _doxid-structsls__control__type_1a184e374e7f9b631289f30f2cbfd4f8a8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` matrix_file_device

specifies the unit number to write the input matrix (in co-ordinate form

.. index:: pair: variable; matrix_file_name
.. _doxid-structsls__control__type_1a0861a0a49e00bd6642d2103f0d5fb7a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char matrix_file_name[31]

name of generated matrix file containing input problem

.. index:: pair: variable; out_of_core_directory
.. _doxid-structsls__control__type_1af13c49b84e4bde8d45c258a969592a1d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_directory[401]

directory name for out of core factorization and additional real workspace in the indefinite case, respectively

.. index:: pair: variable; out_of_core_integer_factor_file
.. _doxid-structsls__control__type_1af99536361792bb2eb73942011fa1c5e9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_integer_factor_file[401]

out of core superfile names for integer and real factor data, real works and additional real workspace in the indefinite case, respectively

.. index:: pair: variable; out_of_core_real_factor_file
.. _doxid-structsls__control__type_1a42bdeec279b8e8856ddc5fc64254b93d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_real_factor_file[401]

see out_of_core_integer_factor_file

.. index:: pair: variable; out_of_core_real_work_file
.. _doxid-structsls__control__type_1ac677c7831bafaddfc5bf60069fdc3171:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_real_work_file[401]

see out_of_core_integer_factor_file

.. index:: pair: variable; out_of_core_indefinite_file
.. _doxid-structsls__control__type_1a6e9acddbc56a33e87fb4532349da129b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_indefinite_file[401]

see out_of_core_integer_factor_file

.. index:: pair: variable; out_of_core_restart_file
.. _doxid-structsls__control__type_1a24a37f3b4cc3a36dfa2954f906a45e11:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char out_of_core_restart_file[501]

see out_of_core_integer_factor_file

.. index:: pair: variable; prefix
.. _doxid-structsls__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

