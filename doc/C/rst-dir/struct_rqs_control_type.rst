.. index:: pair: table; rqs_control_type
.. _doxid-structrqs__control__type:

rqs_control_type structure
--------------------------

.. toctree::
	:hidden:

.. _details-structrqs__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: table; rqs_control_type
.. _doxid-structrqs__control__type:

table rqs_control_type
======================


.. toctree::
	:hidden:

Overview
~~~~~~~~

control derived type as a C struct :ref:`More...<details-structrqs__control__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	rqs_control_type = {
		-- fields
	
		:ref:`f_indexing<doxid-structrqs__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`,
		:ref:`error<doxid-structrqs__control__type_1a11614f44ef4d939bdd984953346a7572>`,
		:ref:`out<doxid-structrqs__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`,
		:ref:`problem<doxid-structrqs__control__type_1a540c0b4e7d398c31890f62ad69cd551c>`,
		:ref:`print_level<doxid-structrqs__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`,
		:ref:`dense_factorization<doxid-structrqs__control__type_1aab4d800411bc0d93a4025eb9e3b863d2>`,
		:ref:`new_h<doxid-structrqs__control__type_1ae60c5b5b987dd62f25253ba4164813f5>`,
		:ref:`new_m<doxid-structrqs__control__type_1a5b8ebe6e4189c3a8d7a0c02acdb21166>`,
		:ref:`new_a<doxid-structrqs__control__type_1a7bea45d51fd9384037bbbf82f7750ce6>`,
		:ref:`max_factorizations<doxid-structrqs__control__type_1a49cdbb7627ab58da229da6ccb3034bb7>`,
		:ref:`inverse_itmax<doxid-structrqs__control__type_1a2ae9a03c4071d26be0d495c9f91f3d45>`,
		:ref:`taylor_max_degree<doxid-structrqs__control__type_1a338fa3956816be173e13bfd4198c4078>`,
		:ref:`initial_multiplier<doxid-structrqs__control__type_1ae8d08df3ba4988681cb5f7c33a20f287>`,
		:ref:`lower<doxid-structrqs__control__type_1a965ee2cfb38687d6f158d35586595eed>`,
		:ref:`upper<doxid-structrqs__control__type_1ab8b6572a40141ada6d5f0455eb806d41>`,
		:ref:`stop_normal<doxid-structrqs__control__type_1a3573530258a38cc836b106b9f7a54565>`,
		:ref:`stop_hard<doxid-structrqs__control__type_1a9508356d815ae3f8eea0f0770fddb6d7>`,
		:ref:`start_invit_tol<doxid-structrqs__control__type_1aec94d12f2b37930ecfdb129e5c4d432d>`,
		:ref:`start_invitmax_tol<doxid-structrqs__control__type_1a75ff746a88cecc883d73cec9c7193bbd>`,
		:ref:`use_initial_multiplier<doxid-structrqs__control__type_1a4d2667d00744ca0f4cc3a2e19bfaae17>`,
		:ref:`initialize_approx_eigenvector<doxid-structrqs__control__type_1a39433cce74413f6635c587d6c06b9110>`,
		:ref:`space_critical<doxid-structrqs__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`,
		:ref:`deallocate_error_fatal<doxid-structrqs__control__type_1a58a2c67fad6e808e8365eff67700cba5>`,
		:ref:`problem_file<doxid-structrqs__control__type_1afbe46916454c2158f31d64ad8dbeaf34>`,
		:ref:`symmetric_linear_solver<doxid-structrqs__control__type_1af297ace351b9307640715643cde57384>`,
		:ref:`definite_linear_solver<doxid-structrqs__control__type_1a9b46b7a8e0af020499e645bef711f634>`,
		:ref:`prefix<doxid-structrqs__control__type_1a1dc05936393ba705f516a0c275df4ffc>`,
		:ref:`sls_control<doxid-structrqs__control__type_1a31b308b91955ee385daacc3de00f161b>`,
		:ref:`ir_control<doxid-structrqs__control__type_1ab87f601227d3bf99916ff3caa3413404>`,
	}

.. _details-structrqs__control__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

control derived type as a C struct

Fields
------

.. index:: pair: variable; f_indexing
.. _doxid-structrqs__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structrqs__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structrqs__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	out

unit for monitor output

.. index:: pair: variable; problem
.. _doxid-structrqs__control__type_1a540c0b4e7d398c31890f62ad69cd551c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	problem

unit to write problem data into file problem_file

.. index:: pair: variable; print_level
.. _doxid-structrqs__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	print_level

controls level of diagnostic output

.. index:: pair: variable; dense_factorization
.. _doxid-structrqs__control__type_1aab4d800411bc0d93a4025eb9e3b863d2:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	dense_factorization

should the problem be solved by dense factorization? Possible values are

* 0 sparse factorization will be used

* 1 dense factorization will be used

* other the choice is made automatically depending on the dimension and sparsity

.. index:: pair: variable; new_h
.. _doxid-structrqs__control__type_1ae60c5b5b987dd62f25253ba4164813f5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	new_h

how much of :math:`H` has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_m
.. _doxid-structrqs__control__type_1a5b8ebe6e4189c3a8d7a0c02acdb21166:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	new_m

how much of :math:`M` has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_a
.. _doxid-structrqs__control__type_1a7bea45d51fd9384037bbbf82f7750ce6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	new_a

how much of :math:`A` has changed since the previous call. Possible values are 0 unchanged 1 values but not indices have changed 2 values and indices have changed

.. index:: pair: variable; max_factorizations
.. _doxid-structrqs__control__type_1a49cdbb7627ab58da229da6ccb3034bb7:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	max_factorizations

the maximum number of factorizations (=iterations) allowed. -ve implies no limit

.. index:: pair: variable; inverse_itmax
.. _doxid-structrqs__control__type_1a2ae9a03c4071d26be0d495c9f91f3d45:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	inverse_itmax

the number of inverse iterations performed in the "maybe hard" case

.. index:: pair: variable; taylor_max_degree
.. _doxid-structrqs__control__type_1a338fa3956816be173e13bfd4198c4078:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	taylor_max_degree

maximum degree of Taylor approximant allowed

.. index:: pair: variable; initial_multiplier
.. _doxid-structrqs__control__type_1ae8d08df3ba4988681cb5f7c33a20f287:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	initial_multiplier

initial estimate of the Lagrange multipler

.. index:: pair: variable; lower
.. _doxid-structrqs__control__type_1a965ee2cfb38687d6f158d35586595eed:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structrqs__control__type_1ab8b6572a40141ada6d5f0455eb806d41:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structrqs__control__type_1a3573530258a38cc836b106b9f7a54565:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	stop_normal

stop when :math:`| \|x\| - (multiplier/\sigma)^(1/(p-2)) | \leq` stop_normal * max :math:`( \|x\|, (multiplier/\sigma)^(1/(p-2)) )` REAL ( KIND = wp ) :: stop_normal = epsmch ** 0.75

.. index:: pair: variable; stop_hard
.. _doxid-structrqs__control__type_1a9508356d815ae3f8eea0f0770fddb6d7:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	stop_hard

stop when bracket on optimal multiplier <= stop_hard * max( bracket ends ) REAL ( KIND = wp ) :: stop_hard = epsmch ** 0.75

.. index:: pair: variable; start_invit_tol
.. _doxid-structrqs__control__type_1aec94d12f2b37930ecfdb129e5c4d432d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	start_invit_tol

start inverse iteration when bracket on optimal multiplier <= stop_start_invit_tol * max( bracket ends )

.. index:: pair: variable; start_invitmax_tol
.. _doxid-structrqs__control__type_1a75ff746a88cecc883d73cec9c7193bbd:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	start_invitmax_tol

start full inverse iteration when bracket on multiplier <= stop_start_invitmax_tol * max( bracket ends)

.. index:: pair: variable; use_initial_multiplier
.. _doxid-structrqs__control__type_1a4d2667d00744ca0f4cc3a2e19bfaae17:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	use_initial_multiplier

ignore initial_multiplier?

.. index:: pair: variable; initialize_approx_eigenvector
.. _doxid-structrqs__control__type_1a39433cce74413f6635c587d6c06b9110:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	initialize_approx_eigenvector

should a suitable initial eigenvector should be chosen or should a previous eigenvector may be used?

.. index:: pair: variable; space_critical
.. _doxid-structrqs__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structrqs__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; problem_file
.. _doxid-structrqs__control__type_1afbe46916454c2158f31d64ad8dbeaf34:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	problem_file

name of file into which to write problem data

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structrqs__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	symmetric_linear_solver

symmetric (indefinite) linear equation solver

.. index:: pair: variable; definite_linear_solver
.. _doxid-structrqs__control__type_1a9b46b7a8e0af020499e645bef711f634:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	definite_linear_solver

definite linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structrqs__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structrqs__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sls_control

control parameters for the Cholesky factorization and solution (see sls_c documentation)

.. index:: pair: variable; ir_control
.. _doxid-structrqs__control__type_1ab87f601227d3bf99916ff3caa3413404:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	ir_control

control parameters for iterative refinement (see ir_c documentation)

