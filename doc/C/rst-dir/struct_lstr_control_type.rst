.. index:: pair: table; lstr_control_type
.. _doxid-structlstr__control__type:

lstr_control_type structure
--------------------------

.. toctree::
	:hidden:

.. _details-structlstr__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: table; lstr_control_type
.. _doxid-structlstr__control__type:

table lstr_control_type
=======================


.. toctree::
	:hidden:

Overview
~~~~~~~~

control derived type as a C struct :ref:`More...<details-structlstr__control__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	lstr_control_type = {
		-- fields
	
		:ref:`f_indexing<doxid-structlstr__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`,
		:ref:`error<doxid-structlstr__control__type_1a11614f44ef4d939bdd984953346a7572>`,
		:ref:`out<doxid-structlstr__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`,
		:ref:`print_level<doxid-structlstr__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`,
		:ref:`start_print<doxid-structlstr__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`,
		:ref:`stop_print<doxid-structlstr__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`,
		:ref:`print_gap<doxid-structlstr__control__type_1a31edaef6b722ef2721633484405a649b>`,
		:ref:`itmin<doxid-structlstr__control__type_1aa385135920896920e910796e164637eb>`,
		:ref:`itmax<doxid-structlstr__control__type_1ac8da2a7f67eddd46d6b08817471e3063>`,
		:ref:`itmax_on_boundary<doxid-structlstr__control__type_1a884ad04ad051e3dbe3d05f56c8b411c8>`,
		:ref:`bitmax<doxid-structlstr__control__type_1a3a99fb44bd37e908c09195de8dc8e455>`,
		:ref:`extra_vectors<doxid-structlstr__control__type_1ac24a274f1682ee791e15979f6c4341e1>`,
		:ref:`stop_relative<doxid-structlstr__control__type_1ae3103abf29cabc33010d53428da2f2fc>`,
		:ref:`stop_absolute<doxid-structlstr__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6>`,
		:ref:`fraction_opt<doxid-structlstr__control__type_1a3a722628453f92a1fb510e15f0bd71bb>`,
		:ref:`time_limit<doxid-structlstr__control__type_1a935b207da67876f712cee43d1e055d75>`,
		:ref:`steihaug_toint<doxid-structlstr__control__type_1a33316102c81d1b21da04eb70850aae95>`,
		:ref:`space_critical<doxid-structlstr__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`,
		:ref:`deallocate_error_fatal<doxid-structlstr__control__type_1a58a2c67fad6e808e8365eff67700cba5>`,
		:ref:`prefix<doxid-structlstr__control__type_1a1dc05936393ba705f516a0c275df4ffc>`,
	}

.. _details-structlstr__control__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

control derived type as a C struct

Fields
------

.. index:: pair: variable; f_indexing
.. _doxid-structlstr__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlstr__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlstr__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlstr__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structlstr__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlstr__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structlstr__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	print_gap

the number of iterations between printing

.. index:: pair: variable; itmin
.. _doxid-structlstr__control__type_1aa385135920896920e910796e164637eb:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	itmin

the minimum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax
.. _doxid-structlstr__control__type_1ac8da2a7f67eddd46d6b08817471e3063:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax_on_boundary
.. _doxid-structlstr__control__type_1a884ad04ad051e3dbe3d05f56c8b411c8:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	itmax_on_boundary

the maximum number of iterations allowed once the boundary has been encountered (-ve = no bound)

.. index:: pair: variable; bitmax
.. _doxid-structlstr__control__type_1a3a99fb44bd37e908c09195de8dc8e455:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bitmax

the maximum number of Newton inner iterations per outer iteration allowe (-ve = no bound)

.. index:: pair: variable; extra_vectors
.. _doxid-structlstr__control__type_1ac24a274f1682ee791e15979f6c4341e1:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; stop_relative
.. _doxid-structlstr__control__type_1ae3103abf29cabc33010d53428da2f2fc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	stop_relative

the iteration stops successfully when :math:`\|A^Tr\|` is less than max( stop_relative * :math:`\|A^Tr_{initial} \|`, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structlstr__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structlstr__control__type_1a3a722628453f92a1fb510e15f0bd71bb:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; time_limit
.. _doxid-structlstr__control__type_1a935b207da67876f712cee43d1e055d75:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time_limit

the maximum elapsed time allowed (-ve means infinite)

.. index:: pair: variable; steihaug_toint
.. _doxid-structlstr__control__type_1a33316102c81d1b21da04eb70850aae95:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	steihaug_toint

should the iteration stop when the Trust-region is first encountered?

.. index:: pair: variable; space_critical
.. _doxid-structlstr__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlstr__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structlstr__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

