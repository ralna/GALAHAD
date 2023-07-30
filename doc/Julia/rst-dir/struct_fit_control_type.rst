.. index:: pair: table; fit_control_type
.. _doxid-structfit__control__type:

fit_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	#include <galahad_fit.h>
	
	struct fit_control_type {
		// fields
	
		Bool :ref:`f_indexing<doxid-structfit__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		Int32 :ref:`error<doxid-structfit__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		Int32 :ref:`out<doxid-structfit__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		Int32 :ref:`print_level<doxid-structfit__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		Bool :ref:`space_critical<doxid-structfit__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		Bool :ref:`deallocate_error_fatal<doxid-structfit__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structfit__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structfit__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structfit__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structfit__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structfit__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structfit__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 print_level

the level of output required is specified by print_level

.. index:: pair: variable; space_critical
.. _doxid-structfit__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation times

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structfit__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structfit__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

