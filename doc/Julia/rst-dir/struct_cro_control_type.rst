.. index:: pair: struct; cro_control_type
.. _doxid-structcro__control__type:

cro_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct cro_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          max_schur_complement::INT
          infinity::T
          feasibility_tolerance::T
          check_io::Bool
          refine_solution::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          symmetric_linear_solver::NTuple{31,Cchar}
          unsymmetric_linear_solver::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          sls_control::sls_control_type{T,INT}
          sbls_control::sbls_control_type{T,INT}
          uls_control::uls_control_type{T,INT}
          ir_control::ir_control_type{T,INT}

.. _details-structcro__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structcro__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structcro__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structcro__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structcro__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; max_schur_complement
.. _doxid-structcro__control__type_max_schur_complement:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_schur_complement

the maximum permitted size of the Schur complement before a refactorization is performed

.. index:: pair: variable; infinity
.. _doxid-structcro__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; feasibility_tolerance
.. _doxid-structcro__control__type_feasibility_tolerance:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T feasibility_tolerance

feasibility tolerance for KKT violation

.. index:: pair: variable; check_io
.. _doxid-structcro__control__type_check_io:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool check_io

if .check_io is true, the input (x,y,z) will be fully tested for consistency

.. index:: pair: variable; refine_solution
.. _doxid-structcro__control__type_refine_solution:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool refine_solution

if .refine solution is true, attempt to satisfy the KKT conditions as accurately as possible

.. index:: pair: variable; space_critical
.. _doxid-structcro__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structcro__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structcro__control__type_symmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

indefinite linear equation solver

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structcro__control__type_unsymmetric_linear_solver:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

unsymmetric linear equation solver

.. index:: pair: variable; prefix
.. _doxid-structcro__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sls_control
.. _doxid-structcro__control__type_sls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; sbls_control
.. _doxid-structcro__control__type_sbls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; uls_control
.. _doxid-structcro__control__type_uls_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`uls_control_type<doxid-structuls__control__type>` uls_control

control parameters for ULS

.. index:: pair: variable; ir_control
.. _doxid-structcro__control__type_ir_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ir_control_type<doxid-structir__control__type>` ir_control

control parameters for iterative refinement

