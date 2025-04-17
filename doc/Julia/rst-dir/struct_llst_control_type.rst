.. index:: pair: table; llst_control_type
.. _doxid-structllst__control__type:

llst_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct llst_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          new_a::INT
          new_s::INT
          max_factorizations::INT
          taylor_max_degree::INT
          initial_multiplier::T
          lower::T
          upper::T
          stop_normal::T
          equality_problem::Bool
          use_initial_multiplier::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          definite_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sbls_control::sbls_control_type{T,INT}
          sls_control::sls_control_type{T,INT}
          ir_control::ir_control_type{T,INT}

.. _details-structllst__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structllst__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structllst__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structllst__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structllst__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; new_a
.. _doxid-structllst__control__type_new_a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_a

how much of $A$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; new_s
.. _doxid-structllst__control__type_new_s:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_s

how much of $S$ has changed since the previous call. Possible values are

* 0 unchanged

* 1 values but not indices have changed

* 2 values and indices have changed

.. index:: pair: variable; max_factorizations
.. _doxid-structllst__control__type_max_factorizations:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_factorizations

the maximum number of factorizations (=iterations) allowed. -ve implies no limit

.. index:: pair: variable; taylor_max_degree
.. _doxid-structllst__control__type_taylor_max_degree:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT taylor_max_degree

maximum degree of Taylor approximant allowed (<= 3)

.. index:: pair: variable; initial_multiplier
.. _doxid-structllst__control__type_initial_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T initial_multiplier

initial estimate of the Lagrange multipler

.. index:: pair: variable; lower
.. _doxid-structllst__control__type_lower:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lower

lower and upper bounds on the multiplier, if known

.. index:: pair: variable; upper
.. _doxid-structllst__control__type_upper:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T upper

see lower

.. index:: pair: variable; stop_normal
.. _doxid-structllst__control__type_stop_normal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_normal

stop when $| \|x\| -$ radius $| \leq$ max( stop_normal \* max( 1, radius )

.. index:: pair: variable; equality_problem
.. _doxid-structllst__control__type_equality_problem:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool equality_problem

is the solution is <b<required to lie on the boundary (i.e., is the constraint an equality)?

.. index:: pair: variable; use_initial_multiplier
.. _doxid-structllst__control__type_use_initial_multiplier:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool use_initial_multiplier

ignore initial_multiplier?

.. index:: pair: variable; space_critical
.. _doxid-structllst__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structllst__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; definite_linear_solver
.. _doxid-structllst__control__type_definite_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

definite linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structllst__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structllst__control__type_sbls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_control
.. _doxid-structllst__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_control
.. _doxid-structllst__control__type_ir_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement for definite system solves (see ir_c documentation)

