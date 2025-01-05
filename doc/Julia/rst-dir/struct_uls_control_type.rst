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
.. _doxid-structuls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structuls__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; warning
.. _doxid-structuls__control__type_1af44472f3bdd4c3af6ed5465266b25312:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT warning

unit for warning messages

.. index:: pair: variable; out
.. _doxid-structuls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structuls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; print_level_solver
.. _doxid-structuls__control__type_1afc17ef86601030a3cb1c7edb7a79a39f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level_solver

controls level of diagnostic output from external solver

.. index:: pair: variable; initial_fill_in_factor
.. _doxid-structuls__control__type_1a9a5f5bbcbc09508812a16fee01fc812d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT initial_fill_in_factor

prediction of factor by which the fill-in will exceed the initial number of nonzeros in $A$

.. index:: pair: variable; min_real_factor_size
.. _doxid-structuls__control__type_1a60c3d0376bbe5c06260509a9be31c562:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT min_real_factor_size

initial size for real array for the factors and other data

.. index:: pair: variable; min_integer_factor_size
.. _doxid-structuls__control__type_1addaddc306c33f5d7e6ebf7eeeab2612e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT min_integer_factor_size

initial size for integer array for the factors and other data

.. index:: pair: variable; max_factor_size
.. _doxid-structuls__control__type_1ae7e9983ad0ee9b1e837f3e1cacc9f4e9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_factor_size

maximum size for real array for the factors and other data

.. index:: pair: variable; blas_block_size_factorize
.. _doxid-structuls__control__type_1aa2177c655d92533f17fcff58482e52c1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT blas_block_size_factorize

level 3 blocking in factorize

.. index:: pair: variable; blas_block_size_solve
.. _doxid-structuls__control__type_1a1af3c20cdeaaee431f309395f9f09564:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT blas_block_size_solve

level 2 and 3 blocking in solve

.. index:: pair: variable; pivot_control
.. _doxid-structuls__control__type_1a7489428a42ce1420b4891f638153c99f:

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
.. _doxid-structuls__control__type_1a9a4b5f33dbbc1f6e9aa81dee63af5d2e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivot_search_limit

number of rows/columns pivot selection restricted to (0 = no restriction)

.. index:: pair: variable; minimum_size_for_btf
.. _doxid-structuls__control__type_1ac1bd945b4d891a5f407f98afccb2c357:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT minimum_size_for_btf

the minimum permitted size of blocks within the block-triangular form

.. index:: pair: variable; max_iterative_refinements
.. _doxid-structuls__control__type_1ab044fd598767830ddc06560a91b80936:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_iterative_refinements

maximum number of iterative refinements allowed

.. index:: pair: variable; stop_if_singular
.. _doxid-structuls__control__type_1a5d1b79218a7bf1f16c3195835a311195:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool stop_if_singular

stop if the matrix is found to be structurally singular

.. index:: pair: variable; array_increase_factor
.. _doxid-structuls__control__type_1a6189cbfe4360772c9fa6b554e93a9b2b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T array_increase_factor

factor by which arrays sizes are to be increased if they are too small

.. index:: pair: variable; switch_to_full_code_density
.. _doxid-structuls__control__type_1a95f87bc195563b7a846fd33107fbe09c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T switch_to_full_code_density

switch to full code when the density exceeds this factor

.. index:: pair: variable; array_decrease_factor
.. _doxid-structuls__control__type_1abee09e4efae03be7d0d1a8503b338ce7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T array_decrease_factor

if previously allocated internal workspace arrays are greater than array_decrease_factor times the currently required sizes, they are reset to current requirements

.. index:: pair: variable; relative_pivot_tolerance
.. _doxid-structuls__control__type_1a65344e4192516e9b621cc7416f09045c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T relative_pivot_tolerance

pivot threshold

.. index:: pair: variable; absolute_pivot_tolerance
.. _doxid-structuls__control__type_1aa5e25bdcf567fac1fb496199a7c06d5a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T absolute_pivot_tolerance

any pivot small than this is considered zero

.. index:: pair: variable; zero_tolerance
.. _doxid-structuls__control__type_1afb83ea8401da6498362914aa88ae823f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_tolerance

any entry smaller than this in modulus is reset to zero

.. index:: pair: variable; acceptable_residual_relative
.. _doxid-structuls__control__type_1a97a6571829dbdccad7598f7b5c3ddfbd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T acceptable_residual_relative

refinement will cease as soon as the residual $\|Ax-b\|$ falls below max( acceptable_residual_relative \* $\|b\|$, acceptable_residual_absolute )

.. index:: pair: variable; acceptable_residual_absolute
.. _doxid-structuls__control__type_1a5ee0e70d90b1398019054b19b68057a0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T acceptable_residual_absolute

see acceptable_residual_relative

.. index:: pair: variable; prefix
.. _doxid-structuls__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

