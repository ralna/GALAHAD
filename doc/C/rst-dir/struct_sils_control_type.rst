.. index:: pair: table; sils_control_type
.. _doxid-structsils__control__type:

sils_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_sils.h>
	
	struct sils_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structsils__control__type_f_indexing>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ICNTL<doxid-structsils__control__type_ICNTL>`[30];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lp<doxid-structsils__control__type_lp>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`wp<doxid-structsils__control__type_wp>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mp<doxid-structsils__control__type_mp>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sp<doxid-structsils__control__type_sp>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ldiag<doxid-structsils__control__type_ldiag>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la<doxid-structsils__control__type_la>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`liw<doxid-structsils__control__type_liw>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxla<doxid-structsils__control__type_maxla>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxliw<doxid-structsils__control__type_maxliw>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pivoting<doxid-structsils__control__type_pivoting>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nemin<doxid-structsils__control__type_nemin>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factorblocking<doxid-structsils__control__type_factorblocking>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`solveblocking<doxid-structsils__control__type_solveblocking>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`thresh<doxid-structsils__control__type_thresh>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ordering<doxid-structsils__control__type_ordering>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`scaling<doxid-structsils__control__type_scaling>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`CNTL<doxid-structsils__control__type_CNTL>`[5];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structsils__control__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce<doxid-structsils__control__type_reduce>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`u<doxid-structsils__control__type_u>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_tolerance<doxid-structsils__control__type_static_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`static_level<doxid-structsils__control__type_static_level>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tolerance<doxid-structsils__control__type_tolerance>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`convergence<doxid-structsils__control__type_convergence>`;
	};
.. _details-structsils__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structsils__control__type_f_indexing:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; ICNTL
.. _doxid-structsils__control__type_ICNTL:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ICNTL[30]

MA27 internal integer controls.

.. index:: pair: variable; lp
.. _doxid-structsils__control__type_lp:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lp

Unit for error messages.

.. index:: pair: variable; wp
.. _doxid-structsils__control__type_wp:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` wp

Unit for warning messages.

.. index:: pair: variable; mp
.. _doxid-structsils__control__type_mp:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mp

Unit for monitor output.

.. index:: pair: variable; sp
.. _doxid-structsils__control__type_sp:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sp

Unit for statistical output.

.. index:: pair: variable; ldiag
.. _doxid-structsils__control__type_ldiag:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ldiag

Controls level of diagnostic output.

.. index:: pair: variable; la
.. _doxid-structsils__control__type_la:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la

Initial size for real array for the factors. If less than nrlnec, default size used.

.. index:: pair: variable; liw
.. _doxid-structsils__control__type_liw:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` liw

Initial size for integer array for the factors. If less than nirnec, default size used.

.. index:: pair: variable; maxla
.. _doxid-structsils__control__type_maxla:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxla

Max. size for real array for the factors.

.. index:: pair: variable; maxliw
.. _doxid-structsils__control__type_maxliw:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxliw

Max. size for integer array for the factors.

.. index:: pair: variable; pivoting
.. _doxid-structsils__control__type_pivoting:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pivoting

Controls pivoting. Possible values are:

* 1 Numerical pivoting will be performed.

* 2 No pivoting will be performed and an error exit will occur immediately a pivot sign change is detected.

* 3 No pivoting will be performed and an error exit will occur if a zero pivot is detected.

* 4 No pivoting is performed but pivots are changed to all be positive.

.. index:: pair: variable; nemin
.. _doxid-structsils__control__type_nemin:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nemin

Minimum number of eliminations in a step (unused)

.. index:: pair: variable; factorblocking
.. _doxid-structsils__control__type_factorblocking:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factorblocking

Level 3 blocking in factorize (unused)

.. index:: pair: variable; solveblocking
.. _doxid-structsils__control__type_solveblocking:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` solveblocking

Level 2 and 3 blocking in solve.

.. index:: pair: variable; thresh
.. _doxid-structsils__control__type_thresh:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` thresh

Controls threshold for detecting full rows in analyse, registered as percentage of N, 100 Only fully dense rows detected (default)

.. index:: pair: variable; ordering
.. _doxid-structsils__control__type_ordering:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ordering

Controls ordering: Possible values are:

* 0 AMD using HSL's MC47

* 1 User defined

* 2 AMD using HSL's MC50

* 3 Min deg as in HSL's MA57

* 4 Metis_nodend ordering

* 5 Ordering chosen depending on matrix characteristics. At the moment choices are HSL's MC50 or Metis_nodend

* >5 Presently equivalent to 5 but may chnage

.. index:: pair: variable; scaling
.. _doxid-structsils__control__type_scaling:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` scaling

Controls scaling: Possible values are:

* 0 No scaling

* >0 Scaling using HSL's MC64 but may change for > 1

.. index:: pair: variable; CNTL
.. _doxid-structsils__control__type_CNTL:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` CNTL[5]

MA27 internal real controls.

.. index:: pair: variable; multiplier
.. _doxid-structsils__control__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

Factor by which arrays sizes are to be increased if they are too small.

.. index:: pair: variable; reduce
.. _doxid-structsils__control__type_reduce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce

If previously allocated internal workspace arrays are greater than reduce times the currently required sizes, they are reset to current requirment.

.. index:: pair: variable; u
.. _doxid-structsils__control__type_u:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` u

Pivot threshold.

.. index:: pair: variable; static_tolerance
.. _doxid-structsils__control__type_static_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_tolerance

used for setting static pivot level

.. index:: pair: variable; static_level
.. _doxid-structsils__control__type_static_level:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` static_level

used for switch to static

.. index:: pair: variable; tolerance
.. _doxid-structsils__control__type_tolerance:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tolerance

Anything less than this is considered zero.

.. index:: pair: variable; convergence
.. _doxid-structsils__control__type_convergence:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` convergence

used to monitor convergence in iterative refinement

