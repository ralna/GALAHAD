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
.. _doxid-structuls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

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
.. _doxid-structuls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; bad_alloc
.. _doxid-structuls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; more_info
.. _doxid-structuls__inform__type_1a24d9e61a5ee1839c2fda4f8d5cff0cb7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more_info

further information on failure

.. index:: pair: variable; out_of_range
.. _doxid-structuls__inform__type_1ae595bb33a1e4f5e95da1927ac4673bbf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structuls__inform__type_1aba4e8149b1ecbb5c21f4534ef36d75b8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 duplicates

number of duplicates

.. index:: pair: variable; entries_dropped
.. _doxid-structuls__inform__type_1a2092110f6de97b4607da184213cd7492:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 entries_dropped

number of entries dropped during the factorization

.. index:: pair: variable; workspace_factors
.. _doxid-structuls__inform__type_1a4e43e0824a84867596aac783665f8057:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 workspace_factors

predicted or actual number of reals and integers to hold factors

.. index:: pair: variable; compresses
.. _doxid-structuls__inform__type_1a06332c9a99b61a82393d2cf74a47428e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT compresses

number of compresses of data required

.. index:: pair: variable; entries_in_factors
.. _doxid-structuls__inform__type_1ab741fb84b79d2b013a84d71932452681:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 entries_in_factors

number of entries in factors

.. index:: pair: variable; rank
.. _doxid-structuls__inform__type_1a6cfd95afd0afebd625b889fb6e58371c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

estimated rank of the matrix

.. index:: pair: variable; structural_rank
.. _doxid-structuls__inform__type_1a870efaeda81975bbbd9e247fe17baed1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT structural_rank

structural rank of the matrix

.. index:: pair: variable; pivot_control
.. _doxid-structuls__inform__type_1a7489428a42ce1420b4891f638153c99f:

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
.. _doxid-structuls__inform__type_1a22c83a8ec8964a169bdd9d0cdf196cf1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iterative_refinements

number of iterative refinements performed

.. index:: pair: variable; alternative
.. _doxid-structuls__inform__type_1a48c07c7da1803ed8af25ca949f4854b5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool alternative

has an "alternative" y: A^T y = 0 and yT b > 0 been found when trying to solve A x = b ?

.. index:: pair: variable; solver
.. _doxid-structuls__inform__type_1af335c33211ea78f2d5b7314d7b1c210d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char solver[21]

name of external solver used to factorize and solve

.. index:: pair: variable; gls_ainfo
.. _doxid-structuls__inform__type_1a8fb46433ac34c44f07902189c1c9d028:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gls_ainfo<doxid-structgls__ainfo>` gls_ainfo

the analyse output structure from gls

.. index:: pair: variable; gls_finfo
.. _doxid-structuls__inform__type_1a8da479a8d3bd8e6b4070e73dc8a3d52a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gls_finfo<doxid-structgls__finfo>` gls_finfo

the factorize output structure from gls

.. index:: pair: variable; gls_sinfo
.. _doxid-structuls__inform__type_1acbf9c9748ab5d9c111befc05c395059a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`gls_sinfo<doxid-structgls__sinfo>` gls_sinfo

the solve output structure from gls

.. index:: pair: variable; ma48_ainfo
.. _doxid-structuls__inform__type_1a9487553043c4ec110826e99e071ce9bc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma48_ainfo<details-structma48__ainfo>` ma48_ainfo

the analyse output structure from hsl_ma48

.. index:: pair: variable; ma48_finfo
.. _doxid-structuls__inform__type_1ac28e039b045cad8dec788fdca9823348:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma48_finfo<details-structma48__finfo>` ma48_finfo

the factorize output structure from hsl_ma48

.. index:: pair: variable; ma48_sinfo
.. _doxid-structuls__inform__type_1a78e9faeec60579cc0fa6c34e0aaeecc3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma48_sinfo<details-structma48__sinfo>` ma48_sinfo

the solve output structure from hsl_ma48

.. index:: pair: variable; lapack_error
.. _doxid-structuls__inform__type_1a0319e2c4ee2d95fe244f92d276038bd4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lapack_error

the LAPACK error return code

