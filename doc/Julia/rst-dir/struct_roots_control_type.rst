.. index:: pair: table; roots_control_type
.. _doxid-structroots__control__type:

roots_control_type structure
----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct roots_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          tol::T
          zero_coef::T
          zero_f::T
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structroots__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structroots__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structroots__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structroots__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structroots__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; tol
.. _doxid-structroots__control__type_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T tol

the required accuracy of the roots

.. index:: pair: variable; zero_coef
.. _doxid-structroots__control__type_zero_coef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_coef

any coefficient smaller in absolute value than zero_coef will be regarde to be zero

.. index:: pair: variable; zero_f
.. _doxid-structroots__control__type_zero_f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_f

any value of the polynomial smaller in absolute value than zero_f will be regarded as giving a root

.. index:: pair: variable; space_critical
.. _doxid-structroots__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structroots__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structroots__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

