.. index:: pair: struct; lpa_control_type
.. _doxid-structlpa__control__type:

lpa_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lpa.h>
	
	struct lpa_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structlpa__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlpa__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlpa__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlpa__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structlpa__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structlpa__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structlpa__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_iterative_refinements<doxid-structlpa__control__type_1ab044fd598767830ddc06560a91b80936>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`min_real_factor_size<doxid-structlpa__control__type_1a60c3d0376bbe5c06260509a9be31c562>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`min_integer_factor_size<doxid-structlpa__control__type_1addaddc306c33f5d7e6ebf7eeeab2612e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`random_number_seed<doxid-structlpa__control__type_1aaaf923bc58fa592eb0fc15779e087eeb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structlpa__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`qplib_file_device<doxid-structlpa__control__type_1a580c343e54a25a2d687782410c9b6917>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structlpa__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tol_data<doxid-structlpa__control__type_1a4464f8bdf548161e0f111713567b93d7>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`feas_tol<doxid-structlpa__control__type_1aa636051ebb14fdebe323a3843e7e24e3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`relative_pivot_tolerance<doxid-structlpa__control__type_1a65344e4192516e9b621cc7416f09045c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`growth_limit<doxid-structlpa__control__type_1a8652f551eab44da378ee44afde39bccb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_tolerance<doxid-structlpa__control__type_1afb83ea8401da6498362914aa88ae823f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`change_tolerance<doxid-structlpa__control__type_1ad16b465967317e22e8290067805720f7>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structlpa__control__type_1abc74ac9bbf6375075f8943aac6ee09e4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structlpa__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structlpa__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`scale<doxid-structlpa__control__type_1aff7a60d3f21b50f4ad18e40d99d33a61>`;
		bool :ref:`dual<doxid-structlpa__control__type_1a7043f852111a1d50ac13fc7c35a6c4f3>`;
		bool :ref:`warm_start<doxid-structlpa__control__type_1a510fc17653a2903edbfeabf9148ff211>`;
		bool :ref:`steepest_edge<doxid-structlpa__control__type_1ac4f367280fb3162d00e79beba8cb7d91>`;
		bool :ref:`space_critical<doxid-structlpa__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structlpa__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`generate_sif_file<doxid-structlpa__control__type_1aa75b3a16d146c0d7ad57bf9817033843>`;
		bool :ref:`generate_qplib_file<doxid-structlpa__control__type_1ad226b26090e539cf47228ba5ec7dc08d>`;
		char :ref:`sif_file_name<doxid-structlpa__control__type_1aaa95e830b709da79d9790471bab54193>`[31];
		char :ref:`qplib_file_name<doxid-structlpa__control__type_1a3d36354e9f30d642f8b081ba85d777d3>`[31];
		char :ref:`prefix<doxid-structlpa__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structlpa__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlpa__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlpa__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlpa__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlpa__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level (>= 2 turns on LA04 output)

.. index:: pair: variable; start_print
.. _doxid-structlpa__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlpa__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structlpa__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; max_iterative_refinements
.. _doxid-structlpa__control__type_1ab044fd598767830ddc06560a91b80936:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_iterative_refinements

maximum number of iterative refinements allowed

.. index:: pair: variable; min_real_factor_size
.. _doxid-structlpa__control__type_1a60c3d0376bbe5c06260509a9be31c562:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` min_real_factor_size

initial size for real array for the factors and other data

.. index:: pair: variable; min_integer_factor_size
.. _doxid-structlpa__control__type_1addaddc306c33f5d7e6ebf7eeeab2612e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` min_integer_factor_size

initial size for integer array for the factors and other data

.. index:: pair: variable; random_number_seed
.. _doxid-structlpa__control__type_1aaaf923bc58fa592eb0fc15779e087eeb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` random_number_seed

the initial seed used when generating random numbers

.. index:: pair: variable; sif_file_device
.. _doxid-structlpa__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; qplib_file_device
.. _doxid-structlpa__control__type_1a580c343e54a25a2d687782410c9b6917:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` qplib_file_device

specifies the unit number to write generated QPLIB file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structlpa__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; tol_data
.. _doxid-structlpa__control__type_1a4464f8bdf548161e0f111713567b93d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tol_data

the tolerable relative perturbation of the data (A,g,..) defining the problem

.. index:: pair: variable; feas_tol
.. _doxid-structlpa__control__type_1aa636051ebb14fdebe323a3843e7e24e3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` feas_tol

any constraint violated by less than feas_tol will be considered to be satisfied

.. index:: pair: variable; relative_pivot_tolerance
.. _doxid-structlpa__control__type_1a65344e4192516e9b621cc7416f09045c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` relative_pivot_tolerance

pivot threshold used to control the selection of pivot elements in the matrix factorization. Any potential pivot which is less than the largest entry in its row times the threshold is excluded as a candidate

.. index:: pair: variable; growth_limit
.. _doxid-structlpa__control__type_1a8652f551eab44da378ee44afde39bccb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` growth_limit

limit to control growth in the upated basis factors. A refactorization occurs if the growth exceeds this limit

.. index:: pair: variable; zero_tolerance
.. _doxid-structlpa__control__type_1afb83ea8401da6498362914aa88ae823f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_tolerance

any entry in the basis smaller than this is considered zero

.. index:: pair: variable; change_tolerance
.. _doxid-structlpa__control__type_1ad16b465967317e22e8290067805720f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` change_tolerance

any solution component whose change is smaller than a tolerence times the largest change may be considered to be zero

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structlpa__control__type_1abc74ac9bbf6375075f8943aac6ee09e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; cpu_time_limit
.. _doxid-structlpa__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structlpa__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; scale
.. _doxid-structlpa__control__type_1aff7a60d3f21b50f4ad18e40d99d33a61:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool scale

if .scale is true, the problem will be automatically scaled prior to solution. This may improve computation time and accuracy

.. index:: pair: variable; dual
.. _doxid-structlpa__control__type_1a7043f852111a1d50ac13fc7c35a6c4f3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool dual

should the dual problem be solved rather than the primal?

.. index:: pair: variable; warm_start
.. _doxid-structlpa__control__type_1a510fc17653a2903edbfeabf9148ff211:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool warm_start

should a warm start using the data in C_stat and X_stat be attempted?

.. index:: pair: variable; steepest_edge
.. _doxid-structlpa__control__type_1ac4f367280fb3162d00e79beba8cb7d91:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool steepest_edge

should steepest-edge weights be used to detetrmine the variable leaving the basis?

.. index:: pair: variable; space_critical
.. _doxid-structlpa__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlpa__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structlpa__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; generate_qplib_file
.. _doxid-structlpa__control__type_1ad226b26090e539cf47228ba5ec7dc08d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_qplib_file

if .generate_qplib_file is .true. if a QPLIB file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structlpa__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name of generated SIF file containing input problem

.. index:: pair: variable; qplib_file_name
.. _doxid-structlpa__control__type_1a3d36354e9f30d642f8b081ba85d777d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char qplib_file_name[31]

name of generated QPLIB file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structlpa__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

