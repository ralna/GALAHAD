.. index:: pair: table; gltr_inform_type
.. _doxid-structgltr__inform__type:

gltr_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_gltr.h>
	
	struct gltr_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structgltr__inform__type_status>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alloc_status<doxid-structgltr__inform__type_alloc_status>`;
		char :ref:`bad_alloc<doxid-structgltr__inform__type_bad_alloc>`[81];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter<doxid-structgltr__inform__type_iter>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`iter_pass2<doxid-structgltr__inform__type_iter_pass2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj<doxid-structgltr__inform__type_obj>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structgltr__inform__type_multiplier>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mnormx<doxid-structgltr__inform__type_mnormx>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`piv<doxid-structgltr__inform__type_piv>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`curv<doxid-structgltr__inform__type_curv>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`rayleigh<doxid-structgltr__inform__type_rayleigh>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`leftmost<doxid-structgltr__inform__type_leftmost>`;
		bool :ref:`negative_curvature<doxid-structgltr__inform__type_negative_curvature>`;
		bool :ref:`hard_case<doxid-structgltr__inform__type_hard_case>`;
	};
.. _details-structgltr__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structgltr__inform__type_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

return status. See :ref:`gltr_solve_problem <doxid-galahad__gltr_8h_1ad77040d245e6bc307d13ea0cec355f18>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structgltr__inform__type_alloc_status:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structgltr__inform__type_bad_alloc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structgltr__inform__type_iter:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structgltr__inform__type_iter_pass2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` iter_pass2

the total number of pass-2 iterations required if the solution lies on the trust-region boundary

.. index:: pair: variable; obj
.. _doxid-structgltr__inform__type_obj:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj

the value of the quadratic function

.. index:: pair: variable; multiplier
.. _doxid-structgltr__inform__type_multiplier:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

the Lagrange multiplier corresponding to the trust-region constraint

.. index:: pair: variable; mnormx
.. _doxid-structgltr__inform__type_mnormx:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mnormx

the $M$ -norm of $x$

.. index:: pair: variable; piv
.. _doxid-structgltr__inform__type_piv:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` piv

the latest pivot in the Cholesky factorization of the Lanczos tridiagona

.. index:: pair: variable; curv
.. _doxid-structgltr__inform__type_curv:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` curv

the most negative cuurvature encountered

.. index:: pair: variable; rayleigh
.. _doxid-structgltr__inform__type_rayleigh:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` rayleigh

the current Rayleigh quotient

.. index:: pair: variable; leftmost
.. _doxid-structgltr__inform__type_leftmost:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` leftmost

an estimate of the leftmost generalized eigenvalue of the pencil $(H,M)$

.. index:: pair: variable; negative_curvature
.. _doxid-structgltr__inform__type_negative_curvature:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool negative_curvature

was negative curvature encountered ?

.. index:: pair: variable; hard_case
.. _doxid-structgltr__inform__type_hard_case:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hard_case

did the hard case occur ?

