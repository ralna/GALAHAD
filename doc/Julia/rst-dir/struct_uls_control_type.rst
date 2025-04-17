.. index:: pair: table; uls_control_type
.. _doxid-structuls__control__type:

uls_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct uls_control_type{T,INT}
          f_indexing::Bool
          error::INT
          warning::INT
          out::INT
          print_level::INT
          print_level_solver::INT
          initial_fill_in_factor::INT
          min_real_factor_size::INT
          min_integer_factor_size::INT
          max_factor_size::Int64
          blas_block_size_factorize::INT
          blas_block_size_solve::INT
          pivot_control::INT
          pivot_search_limit::INT
          minimum_size_for_btf::INT
          max_iterative_refinements::INT
          stop_if_singular::Bool
          array_increase_factor::T
          switch_to_full_code_density::T
          array_decrease_factor::T
          relative_pivot_tolerance::T
          absolute_pivot_tolerance::T
          zero_tolerance::T
          acceptable_residual_relative::T
          acceptable_residual_absolute::T
          prefix::NTuple{31,Cchar}

.. _details-structuls__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structuls__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structuls__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; warning
.. _doxid-structuls__control__type_warning:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT warning

unit for warning messages

.. index:: pair: variable; out
.. _doxid-structuls__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structuls__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; print_level_solver
.. _doxid-structuls__control__type_print_level_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level_solver

controls level of diagnostic output from external solver

.. index:: pair: variable; initial_fill_in_factor
.. _doxid-structuls__control__type_initial_fill_in_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT initial_fill_in_factor

prediction of factor by which the fill-in will exceed the initial number of nonzeros in $A$

.. index:: pair: variable; min_real_factor_size
.. _doxid-structuls__control__type_min_real_factor_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT min_real_factor_size

initial size for real array for the factors and other data

.. index:: pair: variable; min_integer_factor_size
.. _doxid-structuls__control__type_min_integer_factor_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT min_integer_factor_size

initial size for integer array for the factors and other data

.. index:: pair: variable; max_factor_size
.. _doxid-structuls__control__type_max_factor_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_factor_size

maximum size for real array for the factors and other data

.. index:: pair: variable; blas_block_size_factorize
.. _doxid-structuls__control__type_blas_block_size_factorize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT blas_block_size_factorize

level 3 blocking in factorize

.. index:: pair: variable; blas_block_size_solve
.. _doxid-structuls__control__type_blas_block_size_solve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT blas_block_size_solve

level 2 and 3 blocking in solve

.. index:: pair: variable; pivot_control
.. _doxid-structuls__control__type_pivot_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivot_control

pivot control:

* 1 Threshold Partial Pivoting is desired

* 2 Threshold Rook Pivoting is desired

* 3 Threshold Complete Pivoting is desired

* 4 Threshold Symmetric Pivoting is desired

* 5 Threshold Diagonal Pivoting is desired

.. index:: pair: variable; pivot_search_limit
.. _doxid-structuls__control__type_pivot_search_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivot_search_limit

number of rows/columns pivot selection restricted to (0 = no restriction)

.. index:: pair: variable; minimum_size_for_btf
.. _doxid-structuls__control__type_minimum_size_for_btf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT minimum_size_for_btf

the minimum permitted size of blocks within the block-triangular form

.. index:: pair: variable; max_iterative_refinements
.. _doxid-structuls__control__type_max_iterative_refinements:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_iterative_refinements

maximum number of iterative refinements allowed

.. index:: pair: variable; stop_if_singular
.. _doxid-structuls__control__type_stop_if_singular:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool stop_if_singular

stop if the matrix is found to be structurally singular

.. index:: pair: variable; array_increase_factor
.. _doxid-structuls__control__type_array_increase_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T array_increase_factor

factor by which arrays sizes are to be increased if they are too small

.. index:: pair: variable; switch_to_full_code_density
.. _doxid-structuls__control__type_switch_to_full_code_density:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T switch_to_full_code_density

switch to full code when the density exceeds this factor

.. index:: pair: variable; array_decrease_factor
.. _doxid-structuls__control__type_array_decrease_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T array_decrease_factor

if previously allocated internal workspace arrays are greater than array_decrease_factor times the currently required sizes, they are reset to current requirements

.. index:: pair: variable; relative_pivot_tolerance
.. _doxid-structuls__control__type_relative_pivot_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T relative_pivot_tolerance

pivot threshold

.. index:: pair: variable; absolute_pivot_tolerance
.. _doxid-structuls__control__type_absolute_pivot_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T absolute_pivot_tolerance

any pivot small than this is considered zero

.. index:: pair: variable; zero_tolerance
.. _doxid-structuls__control__type_zero_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_tolerance

any entry smaller than this in modulus is reset to zero

.. index:: pair: variable; acceptable_residual_relative
.. _doxid-structuls__control__type_acceptable_residual_relative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T acceptable_residual_relative

refinement will cease as soon as the residual $\|Ax-b\|$ falls below max( acceptable_residual_relative \* $\|b\|$, acceptable_residual_absolute )

.. index:: pair: variable; acceptable_residual_absolute
.. _doxid-structuls__control__type_acceptable_residual_absolute:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T acceptable_residual_absolute

see acceptable_residual_relative

.. index:: pair: variable; prefix
.. _doxid-structuls__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

