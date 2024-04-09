.. index:: pair: table; fdc_control_type
.. _doxid-structfdc__control__type:

fdc_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_fdc.h>
	
	struct fdc_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structfdc__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structfdc__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structfdc__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structfdc__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indmin<doxid-structfdc__control__type_1a5031bbc31f94e4cba6a540a3182b6d80>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`valmin<doxid-structfdc__control__type_1a0e142fa8dc9c363c3c2993b6129b0955>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structfdc__control__type_1a133347eb5f45a24a77b63b4afd4212e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot<doxid-structfdc__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_infeas<doxid-structfdc__control__type_1af32ec1d3b9134c1d7187455c7039dbb1>`;
		bool :ref:`use_sls<doxid-structfdc__control__type_1af0bcd3e9e1917e2e44bb139c9df57e30>`;
		bool :ref:`scale<doxid-structfdc__control__type_1aff7a60d3f21b50f4ad18e40d99d33a61>`;
		bool :ref:`space_critical<doxid-structfdc__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structfdc__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`symmetric_linear_solver<doxid-structfdc__control__type_1af297ace351b9307640715643cde57384>`[31];
		char :ref:`unsymmetric_linear_solver<doxid-structfdc__control__type_1aef6da6b715a0f41983c2a62397104eec>`[31];
		char :ref:`prefix<doxid-structfdc__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structfdc__control__type_1a31b308b91955ee385daacc3de00f161b>`;
		struct :ref:`uls_control_type<doxid-structuls__control__type>` :ref:`uls_control<doxid-structfdc__control__type_1ac6782df4602dd9c04417e2554d72bb00>`;
	};
.. _details-structfdc__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structfdc__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structfdc__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structfdc__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structfdc__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

controls level of diagnostic output

.. index:: pair: variable; indmin
.. _doxid-structfdc__control__type_1a5031bbc31f94e4cba6a540a3182b6d80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indmin

initial estimate of integer workspace for sls (obsolete)

.. index:: pair: variable; valmin
.. _doxid-structfdc__control__type_1a0e142fa8dc9c363c3c2993b6129b0955:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` valmin

initial estimate of real workspace for sls (obsolete)

.. index:: pair: variable; pivot_tol
.. _doxid-structfdc__control__type_1a133347eb5f45a24a77b63b4afd4212e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

the relative pivot tolerance (obsolete)

.. index:: pair: variable; zero_pivot
.. _doxid-structfdc__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot

the absolute pivot tolerance used (obsolete)

.. index:: pair: variable; max_infeas
.. _doxid-structfdc__control__type_1af32ec1d3b9134c1d7187455c7039dbb1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` max_infeas

the largest permitted residual

.. index:: pair: variable; use_sls
.. _doxid-structfdc__control__type_1af0bcd3e9e1917e2e44bb139c9df57e30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool use_sls

choose whether SLS or ULS is used to determine dependencies

.. index:: pair: variable; scale
.. _doxid-structfdc__control__type_1aff7a60d3f21b50f4ad18e40d99d33a61:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool scale

should the rows of A be scaled to have unit infinity norm or should no scaling be applied

.. index:: pair: variable; space_critical
.. _doxid-structfdc__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structfdc__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structfdc__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structfdc__control__type_1aef6da6b715a0f41983c2a62397104eec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

the name of the unsymmetric linear equation solver used. Possible choices are currently: 'gls', 'ma48' and 'getr', although only 'getr' is installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_uls<details-uls__solvers>`.

.. index:: pair: variable; prefix
.. _doxid-structfdc__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structfdc__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; uls_control
.. _doxid-structfdc__control__type_1ac6782df4602dd9c04417e2554d72bb00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`uls_control_type<doxid-structuls__control__type>` uls_control

control parameters for ULS

