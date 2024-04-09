.. index:: pair: struct; bgo_control_type
.. _doxid-structbgo__control__type:

bgo_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bgo.h>
	
	struct bgo_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structbgo__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structbgo__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structbgo__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structbgo__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`attempts_max<doxid-structbgo__control__type_1adf3a400ef30c3d5d65bfc00c68fc291b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_evals<doxid-structbgo__control__type_1a19d3bb811675792cbe138aef2d1c6603>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sampling_strategy<doxid-structbgo__control__type_1a6c37622ea827ff9870202cd50878bda6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`hypercube_discretization<doxid-structbgo__control__type_1a30db27deb26d273fdd69bf125bc86ecd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structbgo__control__type_1a3fc6359d77a53a63d57ea600b51eac13>`;
		char :ref:`alive_file<doxid-structbgo__control__type_1ac631699a26f321b14dbed37115f3c006>`[31];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structbgo__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_unbounded<doxid-structbgo__control__type_1a7eed67e26bc4e17ca334031b7fd608a6>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structbgo__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structbgo__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`random_multistart<doxid-structbgo__control__type_1a172f98defa4da75031c5f280b5cfbab6>`;
		bool :ref:`hessian_available<doxid-structbgo__control__type_1a0fa05e3076ccb30e3b859c1e4be08981>`;
		bool :ref:`space_critical<doxid-structbgo__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structbgo__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structbgo__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`ugo_control_type<doxid-structugo__control__type>` :ref:`ugo_control<doxid-structbgo__control__type_1a750a67a99a91211b1c9521111a471960>`;
		struct :ref:`lhs_control_type<doxid-structlhs__control__type>` :ref:`lhs_control<doxid-structbgo__control__type_1a4938e30d02d3b452486980daef1f6f73>`;
		struct :ref:`trb_control_type<doxid-structtrb__control__type>` :ref:`trb_control<doxid-structbgo__control__type_1a8538960a9c63512c78babb9a8f4b1ca2>`;
	};
.. _details-structbgo__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbgo__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbgo__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structbgo__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structbgo__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* 1 a one-line summary for every improvement

* 2 a summary of each iteration

* $\geq$ 3 increasingly verbose (debugging) output

.. index:: pair: variable; attempts_max
.. _doxid-structbgo__control__type_1adf3a400ef30c3d5d65bfc00c68fc291b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` attempts_max

the maximum number of random searches from the best point found so far

.. index:: pair: variable; max_evals
.. _doxid-structbgo__control__type_1a19d3bb811675792cbe138aef2d1c6603:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_evals

the maximum number of function evaluations made

.. index:: pair: variable; sampling_strategy
.. _doxid-structbgo__control__type_1a6c37622ea827ff9870202cd50878bda6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sampling_strategy

sampling strategy used. Possible values are

* 1 uniformly spread

* 2 Latin hypercube sampling

* 3 niformly spread within a Latin hypercube

.. index:: pair: variable; hypercube_discretization
.. _doxid-structbgo__control__type_1a30db27deb26d273fdd69bf125bc86ecd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` hypercube_discretization

hyper-cube discretization (for sampling stategies 2 and 3)

.. index:: pair: variable; alive_unit
.. _doxid-structbgo__control__type_1a3fc6359d77a53a63d57ea600b51eac13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structbgo__control__type_1ac631699a26f321b14dbed37115f3c006:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; infinity
.. _doxid-structbgo__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; obj_unbounded
.. _doxid-structbgo__control__type_1a7eed67e26bc4e17ca334031b7fd608a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structbgo__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structbgo__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; random_multistart
.. _doxid-structbgo__control__type_1a172f98defa4da75031c5f280b5cfbab6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool random_multistart

perform random-multistart as opposed to local minimize and probe

.. index:: pair: variable; hessian_available
.. _doxid-structbgo__control__type_1a0fa05e3076ccb30e3b859c1e4be08981:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; space_critical
.. _doxid-structbgo__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbgo__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structbgo__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; ugo_control
.. _doxid-structbgo__control__type_1a750a67a99a91211b1c9521111a471960:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`ugo_control_type<doxid-structugo__control__type>` ugo_control

control parameters for UGO

.. index:: pair: variable; lhs_control
.. _doxid-structbgo__control__type_1a4938e30d02d3b452486980daef1f6f73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lhs_control_type<doxid-structlhs__control__type>` lhs_control

control parameters for LHS

.. index:: pair: variable; trb_control
.. _doxid-structbgo__control__type_1a8538960a9c63512c78babb9a8f4b1ca2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`trb_control_type<doxid-structtrb__control__type>` trb_control

control parameters for TRB

