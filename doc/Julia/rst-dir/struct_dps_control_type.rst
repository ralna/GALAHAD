.. index:: pair: table; dps_control_type
.. _doxid-structdps__control__type:

dps_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	#include <galahad_dps.h>
	
	struct dps_control_type {
		// fields
	
		Bool :ref:`f_indexing<doxid-structdps__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		Int32 :ref:`error<doxid-structdps__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		Int32 :ref:`out<doxid-structdps__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		Int32 :ref:`problem<doxid-structdps__control__type_1a540c0b4e7d398c31890f62ad69cd551c>`;
		Int32 :ref:`print_level<doxid-structdps__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		Int32 :ref:`new_h<doxid-structdps__control__type_1ae60c5b5b987dd62f25253ba4164813f5>`;
		Int32 :ref:`taylor_max_degree<doxid-structdps__control__type_1a338fa3956816be173e13bfd4198c4078>`;
		T :ref:`eigen_min<doxid-structdps__control__type_1a133028d7b94139b129034f5b820ffdef>`;
		T :ref:`lower<doxid-structdps__control__type_1a965ee2cfb38687d6f158d35586595eed>`;
		T :ref:`upper<doxid-structdps__control__type_1ab8b6572a40141ada6d5f0455eb806d41>`;
		T :ref:`stop_normal<doxid-structdps__control__type_1a3573530258a38cc836b106b9f7a54565>`;
		T :ref:`stop_absolute_normal<doxid-structdps__control__type_1a02066d2241f2971e375ca4a56532bc2c>`;
		Bool :ref:`goldfarb<doxid-structdps__control__type_1a732b25a17a3b8c219c0a3a948520278c>`;
		Bool :ref:`space_critical<doxid-structdps__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		Bool :ref:`deallocate_error_fatal<doxid-structdps__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`problem_file<doxid-structdps__control__type_1afbe46916454c2158f31d64ad8dbeaf34>`[31];
		char :ref:`symmetric_linear_solver<doxid-structdps__control__type_1af297ace351b9307640715643cde57384>`[31];
		char :ref:`prefix<doxid-structdps__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structdps__control__type_1a31b308b91955ee385daacc3de00f161b>`;
	};
.. _details-structdps__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structdps__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structdps__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structdps__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 out

unit for monitor output

.. index:: pair: variable; problem
.. _doxid-structdps__control__type_1a540c0b4e7d398c31890f62ad69cd551c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 problem

unit to write problem data into file problem_file

.. index:: pair: variable; print_level
.. _doxid-structdps__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 print_level

controls level of diagnostic output

.. index:: pair: variable; new_h
.. _doxid-structdps__control__type_1ae60c5b5b987dd62f25253ba4164813f5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 new_h

how much of :math:`H` has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; taylor_max_degree
.. _doxid-structdps__control__type_1a338fa3956816be173e13bfd4198c4078:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 taylor_max_degree

maximum degree of Taylor approximant allowed

.. index:: pair: variable; eigen_min
.. _doxid-structdps__control__type_1a133028d7b94139b129034f5b820ffdef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T eigen_min

smallest allowable value of an eigenvalue of the block diagonal factor of :math:`H`

.. index:: pair: variable; lower
.. _doxid-structdps__control__type_1a965ee2cfb38687d6f158d35586595eed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structdps__control__type_1ab8b6572a40141ada6d5f0455eb806d41:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structdps__control__type_1a3573530258a38cc836b106b9f7a54565:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_normal

stop trust-region solution when :math:`| ||x||_M - \delta | \leq` max( .stop_normal \* delta, .stop_absolute_normal )

.. index:: pair: variable; stop_absolute_normal
.. _doxid-structdps__control__type_1a02066d2241f2971e375ca4a56532bc2c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_absolute_normal

see stop_normal

.. index:: pair: variable; goldfarb
.. _doxid-structdps__control__type_1a732b25a17a3b8c219c0a3a948520278c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool goldfarb

use the Goldfarb variant of the trust-region/regularization norm rather than the modified absolute-value version

.. index:: pair: variable; space_critical
.. _doxid-structdps__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structdps__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; problem_file
.. _doxid-structdps__control__type_1afbe46916454c2158f31d64ad8dbeaf34:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char problem_file[31]

name of file into which to write problem data

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structdps__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

symmetric (indefinite) linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structdps__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structdps__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the Cholesky factorization and solution

