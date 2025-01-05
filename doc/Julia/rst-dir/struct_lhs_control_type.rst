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
.. _doxid-structlhs__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error.

.. index:: pair: variable; out
.. _doxid-structlhs__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out.

.. index:: pair: variable; print_level
.. _doxid-structlhs__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. Possible values are:

* < 1 no output.

* > 0 debugging.

.. index:: pair: variable; duplication
.. _doxid-structlhs__control__type_1a2c395022ac7da3f286b91b8f79a8edd6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT duplication

the duplication factor. This must be at least 1, a value of 5 is reasonable.

.. index:: pair: variable; space_critical
.. _doxid-structlhs__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time.

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlhs__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue.

.. index:: pair: variable; prefix
.. _doxid-structlhs__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

