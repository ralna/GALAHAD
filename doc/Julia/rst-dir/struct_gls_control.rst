.. index:: pair: struct; gls_control
.. _doxid-structgls__control:

struct gls_control
------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct gls_control_type{T,INT}
          f_indexing::Bool
          lp::INT
          wp::INT
          mp::INT
          ldiag::INT
          btf::INT
          maxit::INT
          factor_blocking::INT
          solve_blas::INT
          la::INT
          la_int::INT
          maxla::INT
          pivoting::INT
          fill_in::INT
          multiplier::T
          reduce::T
          u::T
          switch_full::T
          drop::T
          tolerance::T
          cgce::T
          diagonal_pivoting::Bool
          struct_abort::Bool

.. _details-structgls__control:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structgls__control_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; lp
.. _doxid-structgls__control_lp:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lp

Unit for error messages.

.. index:: pair: variable; wp
.. _doxid-structgls__control_wp:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT wp

Unit for warning messages.

.. index:: pair: variable; mp
.. _doxid-structgls__control_mp:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mp

Unit for monitor output.

.. index:: pair: variable; ldiag
.. _doxid-structgls__control_ldiag:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT ldiag

Controls level of diagnostic output.

.. index:: pair: variable; btf
.. _doxid-structgls__control_1a48e951043b7ec1aab5e2e1b62f1d4021:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT btf

Minimum block size for block-triangular form (BTF). Set to $n$ to avoid.

.. index:: pair: variable; maxit
.. _doxid-structgls__control_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

Maximum number of iterations.

.. index:: pair: variable; factor_blocking
.. _doxid-structgls__control_1abca1e309e7d2d73481534b8fdad872e7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factor_blocking

Level 3 blocking in factorize.

.. index:: pair: variable; solve_blas
.. _doxid-structgls__control_1ab2cc1341183896b96cacc4c6a39acdde:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT solve_blas

Switch for using Level 1 or 2 BLAS in solve.

.. index:: pair: variable; la
.. _doxid-structgls__control_la:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT la

Initial size for real array for the factors.

.. index:: pair: variable; la_int
.. _doxid-structgls__control_1a7d8b6a8163160b2aa8f730ead5dd3727:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT la_int

Initial size for integer array for the factors.

.. index:: pair: variable; maxla
.. _doxid-structgls__control_maxla:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxla

Maximum size for real array for the factors.

.. index:: pair: variable; pivoting
.. _doxid-structgls__control_pivoting:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivoting

Controls pivoting: Number of columns searched. Zero for Markowitz.

.. index:: pair: variable; fill_in
.. _doxid-structgls__control_1afdaf7f9c41586b488e08ac26dd5d93b8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT fill_in

Initially fill_in \* ne space allocated for factors.

.. index:: pair: variable; multiplier
.. _doxid-structgls__control_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

Factor by which arrays sizes are to be increased if they are too small.

.. index:: pair: variable; reduce
.. _doxid-structgls__control_reduce:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reduce

if previously allocated internal workspace arrays are greater than reduce times the currently required sizes, they are reset to current requirment

.. index:: pair: variable; u
.. _doxid-structgls__control_u:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T u

Pivot threshold.

.. index:: pair: variable; switch_full
.. _doxid-structgls__control_1a425fe61f3ab00924d3f1e05e93985df5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T switch_full

Density for switch to full code.

.. index:: pair: variable; drop
.. _doxid-structgls__control_1aa340d2fee9ccb7c4a2f38ac086482506:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T drop

Drop tolerance.

.. index:: pair: variable; tolerance
.. _doxid-structgls__control_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T tolerance

anything < this is considered zero

.. index:: pair: variable; cgce
.. _doxid-structgls__control_1ada0fc18d7059672071369decb96f6178:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cgce

Ratio for required reduction using IR.

.. index:: pair: variable; diagonal_pivoting
.. _doxid-structgls__control_1aae1a90daac2378be15e6d9d644d19281:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool diagonal_pivoting

Set to 0 for diagonal pivoting.

.. index:: pair: variable; struct_abort
.. _doxid-structgls__control_1a9b775b57e2c00f53ca8d8a33e5589533:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool struct_abort

Control to abort if structurally singular.

