.. index:: pair: table; gltr_control_type
.. _doxid-structgltr__control__type:

gltr_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_gltr.h>
	
	struct gltr_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structgltr__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structgltr__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structgltr__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structgltr__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itmax<doxid-structgltr__control__type_1ac8da2a7f67eddd46d6b08817471e3063>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`Lanczos_itmax<doxid-structgltr__control__type_1a414b6b8e330ed0c54599623db9ccb0ba>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`extra_vectors<doxid-structgltr__control__type_1ac24a274f1682ee791e15979f6c4341e1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ritz_printout_device<doxid-structgltr__control__type_1aa733751a194838432e841fce75b56682>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_relative<doxid-structgltr__control__type_1ae3103abf29cabc33010d53428da2f2fc>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_absolute<doxid-structgltr__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`fraction_opt<doxid-structgltr__control__type_1a3a722628453f92a1fb510e15f0bd71bb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`f_min<doxid-structgltr__control__type_1a740cc91342488fc47e0668f1d4ddbbd2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`rminvr_zero<doxid-structgltr__control__type_1a1326abe392007db57e814413298b152f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`f_0<doxid-structgltr__control__type_1a221da497cd332f1acdcdb2919f282fb4>`;
		bool :ref:`unitm<doxid-structgltr__control__type_1ae18a96ff3d3a13fe7a965fc39325d25c>`;
		bool :ref:`steihaug_toint<doxid-structgltr__control__type_1a33316102c81d1b21da04eb70850aae95>`;
		bool :ref:`boundary<doxid-structgltr__control__type_1a68f61635b8b4afe606ebf818e5b4b6d6>`;
		bool :ref:`equality_problem<doxid-structgltr__control__type_1a86fd5b4cf421b63f8d908f27cf2c60bb>`;
		bool :ref:`space_critical<doxid-structgltr__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structgltr__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`print_ritz_values<doxid-structgltr__control__type_1aa12aeab97d40062126b39c7fa300d147>`;
		char :ref:`ritz_file_name<doxid-structgltr__control__type_1afda2047534d65487e814d835cd819316>`[31];
		char :ref:`prefix<doxid-structgltr__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structgltr__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structgltr__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structgltr__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structgltr__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structgltr__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; itmax
.. _doxid-structgltr__control__type_1ac8da2a7f67eddd46d6b08817471e3063:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itmax

the maximum number of iterations allowed (-ve = no bound)

.. index:: pair: variable; Lanczos_itmax
.. _doxid-structgltr__control__type_1a414b6b8e330ed0c54599623db9ccb0ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` Lanczos_itmax

the maximum number of iterations allowed once the boundary has been encountered (-ve = no bound)

.. index:: pair: variable; extra_vectors
.. _doxid-structgltr__control__type_1ac24a274f1682ee791e15979f6c4341e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` extra_vectors

the number of extra work vectors of length n used

.. index:: pair: variable; ritz_printout_device
.. _doxid-structgltr__control__type_1aa733751a194838432e841fce75b56682:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ritz_printout_device

the unit number for writing debug Ritz values

.. index:: pair: variable; stop_relative
.. _doxid-structgltr__control__type_1ae3103abf29cabc33010d53428da2f2fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_relative

the iteration stops successfully when the gradient in the M(inverse) nor is smaller than max( stop_relative \* initial M(inverse) gradient norm, stop_absolute )

.. index:: pair: variable; stop_absolute
.. _doxid-structgltr__control__type_1a16e43fc1e4c1e1b4c671a9b1fbbcd3e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_absolute

see stop_relative

.. index:: pair: variable; fraction_opt
.. _doxid-structgltr__control__type_1a3a722628453f92a1fb510e15f0bd71bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` fraction_opt

an estimate of the solution that gives at least .fraction_opt times the optimal objective value will be found

.. index:: pair: variable; f_min
.. _doxid-structgltr__control__type_1a740cc91342488fc47e0668f1d4ddbbd2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` f_min

the iteration stops if the objective-function value is lower than f_min

.. index:: pair: variable; rminvr_zero
.. _doxid-structgltr__control__type_1a1326abe392007db57e814413298b152f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` rminvr_zero

the smallest value that the square of the M norm of the gradient of the the objective may be before it is considered to be zero

.. index:: pair: variable; f_0
.. _doxid-structgltr__control__type_1a221da497cd332f1acdcdb2919f282fb4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` f_0

the constant term, $f_0$, in the objective function

.. index:: pair: variable; unitm
.. _doxid-structgltr__control__type_1ae18a96ff3d3a13fe7a965fc39325d25c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool unitm

is $M$ the identity matrix ?

.. index:: pair: variable; steihaug_toint
.. _doxid-structgltr__control__type_1a33316102c81d1b21da04eb70850aae95:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool steihaug_toint

should the iteration stop when the Trust-region is first encountered ?

.. index:: pair: variable; boundary
.. _doxid-structgltr__control__type_1a68f61635b8b4afe606ebf818e5b4b6d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool boundary

is the solution thought to lie on the constraint boundary ?

.. index:: pair: variable; equality_problem
.. _doxid-structgltr__control__type_1a86fd5b4cf421b63f8d908f27cf2c60bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool equality_problem

is the solution required to lie on the constraint boundary ?

.. index:: pair: variable; space_critical
.. _doxid-structgltr__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structgltr__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; print_ritz_values
.. _doxid-structgltr__control__type_1aa12aeab97d40062126b39c7fa300d147:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool print_ritz_values

should the Ritz values be written to the debug stream?

.. index:: pair: variable; ritz_file_name
.. _doxid-structgltr__control__type_1afda2047534d65487e814d835cd819316:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char ritz_file_name[31]

name of debug file containing the Ritz values

.. index:: pair: variable; prefix
.. _doxid-structgltr__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

