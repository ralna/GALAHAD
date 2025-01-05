.. index:: pair: table; trs_control_type
.. _doxid-structtrs__control__type:

trs_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct trs_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          problem::INT
          print_level::INT
          dense_factorization::INT
          new_h::INT
          new_m::INT
          new_a::INT
          max_factorizations::INT
          inverse_itmax::INT
          taylor_max_degree::INT
          initial_multiplier::T
          lower::T
          upper::T
          stop_normal::T
          stop_absolute_normal::T
          stop_hard::T
          start_invit_tol::T
          start_invitmax_tol::T
          equality_problem::Bool
          use_initial_multiplier::Bool
          initialize_approx_eigenvector::Bool
          force_Newton::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          problem_file::NTuple{31,Cchar}
          symmetric_linear_solver::NTuple{31,Cchar}
          definite_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
          ir_control::ir_control_type{T,INT}

.. _details-structtrs__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structtrs__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structtrs__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structtrs__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; problem
.. _doxid-structtrs__control__type_1a540c0b4e7d398c31890f62ad69cd551c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT problem

unit to write problem data into file problem_file

.. index:: pair: variable; print_level
.. _doxid-structtrs__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; dense_factorization
.. _doxid-structtrs__control__type_1aab4d800411bc0d93a4025eb9e3b863d2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dense_factorization

should the problem be solved by dense factorization? Possible values are

* 0 sparse factorization will be used

* 1 dense factorization will be used

* other the choice is made automatically depending on the dimension & sparsity

.. index:: pair: variable; new_h
.. _doxid-structtrs__control__type_1ae60c5b5b987dd62f25253ba4164813f5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_h

how much of $H$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_m
.. _doxid-structtrs__control__type_1a5b8ebe6e4189c3a8d7a0c02acdb21166:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_m

how much of $M$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_a
.. _doxid-structtrs__control__type_1a7bea45d51fd9384037bbbf82f7750ce6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_a

how much of $A$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; max_factorizations
.. _doxid-structtrs__control__type_1a49cdbb7627ab58da229da6ccb3034bb7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_factorizations

the maximum number of factorizations (=iterations) allowed. -ve implies no limit

.. index:: pair: variable; inverse_itmax
.. _doxid-structtrs__control__type_1a2ae9a03c4071d26be0d495c9f91f3d45:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT inverse_itmax

the number of inverse iterations performed in the "maybe hard" case

.. index:: pair: variable; taylor_max_degree
.. _doxid-structtrs__control__type_1a338fa3956816be173e13bfd4198c4078:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT taylor_max_degree

maximum degree of Taylor approximant allowed

.. index:: pair: variable; initial_multiplier
.. _doxid-structtrs__control__type_1ae8d08df3ba4988681cb5f7c33a20f287:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_multiplier

initial estimate of the Lagrange multipler

.. index:: pair: variable; lower
.. _doxid-structtrs__control__type_1a965ee2cfb38687d6f158d35586595eed:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structtrs__control__type_1ab8b6572a40141ada6d5f0455eb806d41:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structtrs__control__type_1a3573530258a38cc836b106b9f7a54565:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_normal

stop when $| ||x|| - radius | \leq$ max( stop_normal \* radius, stop_absolute_normal )

.. index:: pair: variable; stop_absolute_normal
.. _doxid-structtrs__control__type_1a02066d2241f2971e375ca4a56532bc2c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_absolute_normal

see stop_normal

.. index:: pair: variable; stop_hard
.. _doxid-structtrs__control__type_1a9508356d815ae3f8eea0f0770fddb6d7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_hard

stop when bracket on optimal multiplier $\leq$ stop_hard \* max( bracket ends )

.. index:: pair: variable; start_invit_tol
.. _doxid-structtrs__control__type_1aec94d12f2b37930ecfdb129e5c4d432d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T start_invit_tol

start inverse iteration when bracket on optimal multiplier $\leq$ stop_start_invit_tol \* max( bracket ends )

.. index:: pair: variable; start_invitmax_tol
.. _doxid-structtrs__control__type_1a75ff746a88cecc883d73cec9c7193bbd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T start_invitmax_tol

start full inverse iteration when bracket on multiplier $\leq$ stop_start_invitmax_tol \* max( bracket ends)

.. index:: pair: variable; equality_problem
.. _doxid-structtrs__control__type_1a86fd5b4cf421b63f8d908f27cf2c60bb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool equality_problem

is the solution is <b<required to lie on the boundary (i.e., is the constraint an equality)?

.. index:: pair: variable; use_initial_multiplier
.. _doxid-structtrs__control__type_1a4d2667d00744ca0f4cc3a2e19bfaae17:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool use_initial_multiplier

ignore initial_multiplier?

.. index:: pair: variable; initialize_approx_eigenvector
.. _doxid-structtrs__control__type_1a39433cce74413f6635c587d6c06b9110:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool initialize_approx_eigenvector

should a suitable initial eigenvector should be chosen or should a previous eigenvector may be used?

.. index:: pair: variable; force_Newton
.. _doxid-structtrs__control__type_1aec07be06c4b1e151259949c82d70c675:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool force_Newton

ignore the trust-region if $H$ is positive definite

.. index:: pair: variable; space_critical
.. _doxid-structtrs__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structtrs__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; problem_file
.. _doxid-structtrs__control__type_1afbe46916454c2158f31d64ad8dbeaf34:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char problem_file[31]

name of file into which to write problem data

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structtrs__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

symmetric (indefinite) linear equation solver

.. index:: pair: variable; definite_linear_solver
.. _doxid-structtrs__control__type_1a9b46b7a8e0af020499e645bef711f634:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

definite linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structtrs__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structtrs__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the Cholesky factorization and solution (see sls_c documentation)

.. index:: pair: variable; ir_control
.. _doxid-structtrs__control__type_1ab87f601227d3bf99916ff3caa3413404:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement (see ir_c documentation)

