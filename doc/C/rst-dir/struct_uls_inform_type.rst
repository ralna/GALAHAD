.. index:: pair: table; uls_inform_type
.. _doxid-structuls__inform__type:

uls_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_uls.h>
	
	struct uls_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structuls__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structuls__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structuls__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`more_info<doxid-structuls__inform__type_more_info>`;
		int64_t :ref:`out_of_range<doxid-structuls__inform__type_out_of_range>`;
		int64_t :ref:`duplicates<doxid-structuls__inform__type_duplicates>`;
		int64_t :ref:`entries_dropped<doxid-structuls__inform__type_entries_dropped>`;
		int64_t :ref:`workspace_factors<doxid-structuls__inform__type_workspace_factors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`compresses<doxid-structuls__inform__type_compresses>`;
		int64_t :ref:`entries_in_factors<doxid-structuls__inform__type_entries_in_factors>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`rank<doxid-structuls__inform__type_rank>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`structural_rank<doxid-structuls__inform__type_structural_rank>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pivot_control<doxid-structuls__inform__type_pivot_control>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iterative_refinements<doxid-structuls__inform__type_iterative_refinements>`;
		bool :ref:`alternative<doxid-structuls__inform__type_alternative>`;
		char :ref:`solver<doxid-structuls__inform__type_solver>`[21];
		struct gls_ainfo_type :ref:`gls_ainfo<doxid-structuls__inform__type_gls_ainfo>`;
		struct gls_finfo_type :ref:`gls_finfo<doxid-structuls__inform__type_gls_finfo>`;
		struct gls_sinfo_type :ref:`gls_sinfo<doxid-structuls__inform__type_gls_sinfo>`;
		struct ma48_ainfo :ref:`ma48_ainfo<doxid-structuls__inform__type_ma48_ainfo>`;
		struct ma48_finfo :ref:`ma48_finfo<doxid-structuls__inform__type_ma48_finfo>`;
		struct ma48_sinfo :ref:`ma48_sinfo<doxid-structuls__inform__type_ma48_sinfo>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lapack_error<doxid-structuls__inform__type_lapack_error>`;
	};
.. _details-structuls__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structuls__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

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

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

STAT value after allocate failure.

.. index:: pair: variable; bad_alloc
.. _doxid-structuls__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

name of array which provoked an allocate failure

.. index:: pair: variable; more_info
.. _doxid-structuls__inform__type_more_info:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` more_info

further information on failure

.. index:: pair: variable; out_of_range
.. _doxid-structuls__inform__type_out_of_range:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structuls__inform__type_duplicates:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t duplicates

number of duplicates

.. index:: pair: variable; entries_dropped
.. _doxid-structuls__inform__type_entries_dropped:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t entries_dropped

number of entries dropped during the factorization

.. index:: pair: variable; workspace_factors
.. _doxid-structuls__inform__type_workspace_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t workspace_factors

predicted or actual number of reals and integers to hold factors

.. index:: pair: variable; compresses
.. _doxid-structuls__inform__type_compresses:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` compresses

number of compresses of data required

.. index:: pair: variable; entries_in_factors
.. _doxid-structuls__inform__type_entries_in_factors:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t entries_in_factors

number of entries in factors

.. index:: pair: variable; rank
.. _doxid-structuls__inform__type_rank:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` rank

estimated rank of the matrix

.. index:: pair: variable; structural_rank
.. _doxid-structuls__inform__type_structural_rank:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` structural_rank

structural rank of the matrix

.. index:: pair: variable; pivot_control
.. _doxid-structuls__inform__type_pivot_control:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pivot_control

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

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iterative_refinements

number of iterative refinements performed

.. index:: pair: variable; alternative
.. _doxid-structuls__inform__type_alternative:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool alternative

has an "alternative" y: A^T y = 0 and yT b > 0 been found when trying to solve A x = b ?

.. index:: pair: variable; solver
.. _doxid-structuls__inform__type_solver:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char solver[21]

name of external solver used to factorize and solve

.. index:: pair: variable; gls_ainfo
.. _doxid-structuls__inform__type_gls_ainfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gls_ainfo_type<doxid-structgls__ainfo>` gls_ainfo

the output arrays from GLS

.. index:: pair: variable; gls_finfo
.. _doxid-structuls__inform__type_gls_finfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gls_finfo_type<doxid-structgls__finfo>` gls_finfo

see gls_ainfo

.. index:: pair: variable; gls_sinfo
.. _doxid-structuls__inform__type_gls_sinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gls_sinfo_type<doxid-structgls__sinfo>` gls_sinfo

see gls_ainfo

.. index:: pair: variable; ma48_ainfo
.. _doxid-structuls__inform__type_ma48_ainfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ma48_ainfo<doxid-structma48__ainfo>` ma48_ainfo

the output arrays from HSL's MA48

.. index:: pair: variable; ma48_finfo
.. _doxid-structuls__inform__type_ma48_finfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ma48_finfo<doxid-structma48__finfo>` ma48_finfo

see ma48_ainfo

.. index:: pair: variable; ma48_sinfo
.. _doxid-structuls__inform__type_ma48_sinfo:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ma48_sinfo<doxid-structma48__sinfo>` ma48_sinfo

see ma48_ainfo

.. index:: pair: variable; lapack_error
.. _doxid-structuls__inform__type_lapack_error:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lapack_error

the LAPACK error return code

