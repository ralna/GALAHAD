.. index:: pair: table; uls_inform_type
.. _doxid-structuls__inform__type:

uls_inform_type structure
-------------------------

.. toctree::
	:hidden:

	struct_hsl_types.rst

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct uls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          more_info::INT
          out_of_range::Int64
          duplicates::Int64
          entries_dropped::Int64
          workspace_factors::Int64
          compresses::INT
          entries_in_factors::Int64
          rank::INT
          structural_rank::INT
          pivot_control::INT
          iterative_refinements::INT
          alternative::Bool
          solver::NTuple{21,Cchar}
          gls_ainfo_type::gls_ainfo_type{T,INT}
          gls_finfo_type::gls_finfo_type{T,INT}
          gls_sinfo_type::gls_sinfo_type{INT}
          ma48_ainfo::ma48_ainfo{T,INT}
          ma48_finfo::ma48_finfo{T,INT}
          ma48_sinfo::ma48_sinfo{INT}
          lapack_error::INT

.. _details-structuls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structuls__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status:

* **0**

  success

* **-1**

  allocation error

* **-2**

  deallocation error

* **-3**

  matrix data faulty (m < 1, n < 1, ne < 0)

* **-26**

  unknown solver

* **-29**

  unavailable option

* **-31**

  input order is not a permutation or is faulty in some other way

* **-32**

  error with integer workspace

* **-33**

  error with real workspace

* **-50**

  solver-specific error; see the solver's info parameter

.. index:: pair: variable; alloc_status
.. _doxid-structuls__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; bad_alloc
.. _doxid-structuls__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; more_info
.. _doxid-structuls__inform__type_more_info:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more_info

further information on failure

.. index:: pair: variable; out_of_range
.. _doxid-structuls__inform__type_out_of_range:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structuls__inform__type_duplicates:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 duplicates

number of duplicates

.. index:: pair: variable; entries_dropped
.. _doxid-structuls__inform__type_entries_dropped:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 entries_dropped

number of entries dropped during the factorization

.. index:: pair: variable; workspace_factors
.. _doxid-structuls__inform__type_workspace_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 workspace_factors

predicted or actual number of reals and integers to hold factors

.. index:: pair: variable; compresses
.. _doxid-structuls__inform__type_compresses:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT compresses

number of compresses of data required

.. index:: pair: variable; entries_in_factors
.. _doxid-structuls__inform__type_entries_in_factors:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 entries_in_factors

number of entries in factors

.. index:: pair: variable; rank
.. _doxid-structuls__inform__type_rank:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

estimated rank of the matrix

.. index:: pair: variable; structural_rank
.. _doxid-structuls__inform__type_structural_rank:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT structural_rank

structural rank of the matrix

.. index:: pair: variable; pivot_control
.. _doxid-structuls__inform__type_pivot_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivot_control

pivot control:

* **1**

  Threshold Partial Pivoting has been used

* **2**

  Threshold Rook Pivoting has been used

* **3**

  Threshold Complete Pivoting has been desired

* **4**

  Threshold Symmetric Pivoting has been desired

* **5**

  Threshold Diagonal Pivoting has been desired

.. index:: pair: variable; iterative_refinements
.. _doxid-structuls__inform__type_iterative_refinements:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iterative_refinements

number of iterative refinements performed

.. index:: pair: variable; alternative
.. _doxid-structuls__inform__type_alternative:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool alternative

has an "alternative" y: A^T y = 0 and yT b > 0 been found when trying to solve A x = b ?

.. index:: pair: variable; solver
.. _doxid-structuls__inform__type_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char solver[21]

name of external solver used to factorize and solve

.. index:: pair: variable; gls_ainfo
.. _doxid-structuls__inform__type_gls_ainfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gls_ainfo<doxid-structgls__ainfo>` gls_ainfo

the analyse output structure from gls

.. index:: pair: variable; gls_finfo
.. _doxid-structuls__inform__type_gls_finfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gls_finfo<doxid-structgls__finfo>` gls_finfo

the factorize output structure from gls

.. index:: pair: variable; gls_sinfo
.. _doxid-structuls__inform__type_gls_sinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gls_sinfo<doxid-structgls__sinfo>` gls_sinfo

the solve output structure from gls

.. index:: pair: variable; ma48_ainfo
.. _doxid-structuls__inform__type_ma48_ainfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma48_ainfo<details-structma48__ainfo>` ma48_ainfo

the analyse output structure from hsl_ma48

.. index:: pair: variable; ma48_finfo
.. _doxid-structuls__inform__type_ma48_finfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma48_finfo<details-structma48__finfo>` ma48_finfo

the factorize output structure from hsl_ma48

.. index:: pair: variable; ma48_sinfo
.. _doxid-structuls__inform__type_ma48_sinfo:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma48_sinfo<details-structma48__sinfo>` ma48_sinfo

the solve output structure from hsl_ma48

.. index:: pair: variable; lapack_error
.. _doxid-structuls__inform__type_lapack_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lapack_error

the LAPACK error return code

