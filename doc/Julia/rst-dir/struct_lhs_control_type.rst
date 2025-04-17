.. index:: pair: table; lhs_control_type
.. _doxid-structlhs__control__type:

lhs_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lhs_control_type{INT}
          error::INT
          out::INT
          print_level::INT
          duplication::INT
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structlhs__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structlhs__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error.

.. index:: pair: variable; out
.. _doxid-structlhs__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out.

.. index:: pair: variable; print_level
.. _doxid-structlhs__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. Possible values are:

* < 1 no output.

* > 0 debugging.

.. index:: pair: variable; duplication
.. _doxid-structlhs__control__type_duplication:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT duplication

the duplication factor. This must be at least 1, a value of 5 is reasonable.

.. index:: pair: variable; space_critical
.. _doxid-structlhs__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time.

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlhs__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue.

.. index:: pair: variable; prefix
.. _doxid-structlhs__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

