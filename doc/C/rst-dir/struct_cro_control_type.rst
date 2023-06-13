.. index:: pair: table; cro_control_type
.. _doxid-structcro__control__type:

cro_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	cro_control_type = {
		-- fields
	
		:ref:`f_indexing<doxid-structcro__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`,
		:ref:`error<doxid-structcro__control__type_1a11614f44ef4d939bdd984953346a7572>`,
		:ref:`out<doxid-structcro__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`,
		:ref:`print_level<doxid-structcro__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`,
		:ref:`max_schur_complement<doxid-structcro__control__type_1abd1f3cb576a120eb097ebb819874af8d>`,
		:ref:`infinity<doxid-structcro__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`,
		:ref:`feasibility_tolerance<doxid-structcro__control__type_1aefac38140eecc872a3bc2907de2f0a30>`,
		:ref:`check_io<doxid-structcro__control__type_1a26ffe1bc01e525bfbc88f91b08e2295d>`,
		:ref:`refine_solution<doxid-structcro__control__type_1a31c847b86043424b65e29784b7196b78>`,
		:ref:`space_critical<doxid-structcro__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`,
		:ref:`deallocate_error_fatal<doxid-structcro__control__type_1a58a2c67fad6e808e8365eff67700cba5>`,
		:ref:`symmetric_linear_solver<doxid-structcro__control__type_1af297ace351b9307640715643cde57384>`,
		:ref:`unsymmetric_linear_solver<doxid-structcro__control__type_1aef6da6b715a0f41983c2a62397104eec>`,
		:ref:`prefix<doxid-structcro__control__type_1a1dc05936393ba705f516a0c275df4ffc>`,
		:ref:`sls_control<doxid-structcro__control__type_1a31b308b91955ee385daacc3de00f161b>`,
		:ref:`sbls_control<doxid-structcro__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a>`,
		:ref:`uls_control<doxid-structcro__control__type_1ac6782df4602dd9c04417e2554d72bb00>`,
		:ref:`ir_control<doxid-structcro__control__type_1ab87f601227d3bf99916ff3caa3413404>`,
	}

.. _details-structcro__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structcro__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structcro__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structcro__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structcro__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	print_level

the level of output required is specified by print_level

.. index:: pair: variable; max_schur_complement
.. _doxid-structcro__control__type_1abd1f3cb576a120eb097ebb819874af8d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	max_schur_complement

the maximum permitted size of the Schur complement before a refactorization is performed

.. index:: pair: variable; infinity
.. _doxid-structcro__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; feasibility_tolerance
.. _doxid-structcro__control__type_1aefac38140eecc872a3bc2907de2f0a30:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	feasibility_tolerance

feasibility tolerance for KKT violation

.. index:: pair: variable; check_io
.. _doxid-structcro__control__type_1a26ffe1bc01e525bfbc88f91b08e2295d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	check_io

if .check_io is true, the input (x,y,z) will be fully tested for consistency

.. index:: pair: variable; refine_solution
.. _doxid-structcro__control__type_1a31c847b86043424b65e29784b7196b78:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	refine_solution

if .refine solution is true, attempt to satisfy the KKT conditions as accurately as possible

.. index:: pair: variable; space_critical
.. _doxid-structcro__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structcro__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structcro__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	symmetric_linear_solver

indefinite linear equation solver

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structcro__control__type_1aef6da6b715a0f41983c2a62397104eec:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	unsymmetric_linear_solver

unsymmetric linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structcro__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structcro__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sls_control

control parameters for SLS

.. index:: pair: variable; sbls_control
.. _doxid-structcro__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sbls_control

control parameters for SBLS

.. index:: pair: variable; uls_control
.. _doxid-structcro__control__type_1ac6782df4602dd9c04417e2554d72bb00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	uls_control

control parameters for ULS

.. index:: pair: variable; ir_control
.. _doxid-structcro__control__type_1ab87f601227d3bf99916ff3caa3413404:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	ir_control

control parameters for iterative refinement

