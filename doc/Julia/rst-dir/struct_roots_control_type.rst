.. index:: pair: table; roots_control_type
.. _doxid-structroots__control__type:

roots_control_type structure
----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

	#include <galahad_roots.h>
	
	struct roots_control_type {
		// fields
	
		Bool :ref:`f_indexing<doxid-structroots__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		Int32 :ref:`error<doxid-structroots__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		Int32 :ref:`out<doxid-structroots__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		Int32 :ref:`print_level<doxid-structroots__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		T :ref:`tol<doxid-structroots__control__type_1adbee5a29b5dbbe5274332980af85f697>`;
		T :ref:`zero_coef<doxid-structroots__control__type_1a2760b013e8ef1a2ab6d0a0301379e10f>`;
		T :ref:`zero_f<doxid-structroots__control__type_1a1c2d0687cfd3cda18d1595b79a7fdfe8>`;
		Bool :ref:`space_critical<doxid-structroots__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		Bool :ref:`deallocate_error_fatal<doxid-structroots__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structroots__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structroots__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structroots__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structroots__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structroots__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structroots__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int32 print_level

the level of output required is specified by print_level

.. index:: pair: variable; tol
.. _doxid-structroots__control__type_1adbee5a29b5dbbe5274332980af85f697:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T tol

the required accuracy of the roots

.. index:: pair: variable; zero_coef
.. _doxid-structroots__control__type_1a2760b013e8ef1a2ab6d0a0301379e10f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_coef

any coefficient smaller in absolute value than zero_coef will be regarde to be zero

.. index:: pair: variable; zero_f
.. _doxid-structroots__control__type_1a1c2d0687cfd3cda18d1595b79a7fdfe8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T zero_f

any value of the polynomial smaller in absolute value than zero_f will be regarded as giving a root

.. index:: pair: variable; space_critical
.. _doxid-structroots__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structroots__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structroots__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

