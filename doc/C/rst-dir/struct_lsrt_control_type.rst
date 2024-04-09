.. index:: pair: table; lsrt_control_type
.. _doxid-structlsrt__control__type:

lsrt_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lsrt.h>
	
	struct lsrt_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structlsrt__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structlsrt__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structlsrt__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structlsrt__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structlsrt__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structlsrt__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structlsrt__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmin<doxid-structlsrt__control__type_1aa385135920896920e910796e164637eb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmax<doxid-structlsrt__control__type_1ac8da2a7f67eddd46d6b08817471e3063>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`bitmax<doxid-structlsrt__control__type_1a3a99fb44bd37e908c09195de8dc8e455>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_vectors<doxid-structlsrt__control__type_1ac24a274f1682ee791e15979f6c4341e1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stopping_rule<doxid-structlsrt__control__type_1a39dcb82333184b2fa9251a89c3e30e1a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`freq<doxid-structlsrt__control__type_1ae0d22272b68e75d19ac0b80c01f806b6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_relative<doxid-structlsrt__control__type_1ae3103abf29cabc33010d53428da2f2fc>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute<doxid-structlsrt__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`fraction_opt<doxid-structlsrt__control__type_1a3a722628453f92a1fb510e15f0bd71bb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`time_limit<doxid-structlsrt__control__type_1a935b207da67876f712cee43d1e055d75>`;
		bool :ref:`space_critical<doxid-structlsrt__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structlsrt__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structlsrt__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structlsrt__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlsrt__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlsrt__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structlsrt__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structlsrt__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structlsrt__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structlsrt__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structlsrt__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; itmin
.. _doxid-structlsrt__control__type_1aa385135920896920e910796e164637eb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmin

the minimum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; itmax
.. _doxid-structlsrt__control__type_1ac8da2a7f67eddd46d6b08817471e3063:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; bitmax
.. _doxid-structlsrt__control__type_1a3a99fb44bd37e908c09195de8dc8e455:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` bitmax

the maximum number of Newton inner iterations per outer iteration allowed (-ve = no bound)

.. index:: pair: variable; extra_vectors
.. _doxid-structlsrt__control__type_1ac24a274f1682ee791e15979f6c4341e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; stopping_rule
.. _doxid-structlsrt__control__type_1a39dcb82333184b2fa9251a89c3e30e1a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stopping_rule

the stopping rule used: 0=1.0, 1=norm step, 2=norm step/sigma (NOT USED)

.. index:: pair: variable; freq
.. _doxid-structlsrt__control__type_1ae0d22272b68e75d19ac0b80c01f806b6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` freq

frequency for solving the reduced tri-diagonal problem (NOT USED)

.. index:: pair: variable; stop_relative
.. _doxid-structlsrt__control__type_1ae3103abf29cabc33010d53428da2f2fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_relative

the iteration stops successfully when \|\|A^Tr\|\| is less than max( stop_relative \* \|\|A^Tr initial \|\|, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structlsrt__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structlsrt__control__type_1a3a722628453f92a1fb510e15f0bd71bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; time_limit
.. _doxid-structlsrt__control__type_1a935b207da67876f712cee43d1e055d75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` time_limit

the maximum elapsed time allowed (-ve means infinite)

.. index:: pair: variable; space_critical
.. _doxid-structlsrt__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlsrt__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structlsrt__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

