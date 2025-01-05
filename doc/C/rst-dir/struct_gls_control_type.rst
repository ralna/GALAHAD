.. index:: pair: struct; gls_control_type
.. _doxid-structgls__control__type:

struct gls_control_type
-----------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_gls.h>
	
	struct gls_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structgls__control_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lp<doxid-structgls__control_1a3eec33a4e6d8295c25d117bb25dd1b9b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`wp<doxid-structgls__control_1af203f5ddbac4a47afed1a07b97e7e477>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`mp<doxid-structgls__control_1a4b5efeeced2b749191f71afc3bc8bebd>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ldiag<doxid-structgls__control_1ad0e905fe282a7125424a53219afc0791>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`btf<doxid-structgls__control_1a48e951043b7ec1aab5e2e1b62f1d4021>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structgls__control_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factor_blocking<doxid-structgls__control_1abca1e309e7d2d73481534b8fdad872e7>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`solve_blas<doxid-structgls__control_1ab2cc1341183896b96cacc4c6a39acdde>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la<doxid-structgls__control_1a8e48dcc59e4b8bbe40fe5b58321e4e72>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`la_int<doxid-structgls__control_1a7d8b6a8163160b2aa8f730ead5dd3727>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxla<doxid-structgls__control_1a6437ead17fd48daf197640949e8d4ff3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`pivoting<doxid-structgls__control_1aa4d2d93f87da5df80b8aa2bce688c030>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`fill_in<doxid-structgls__control_1afdaf7f9c41586b488e08ac26dd5d93b8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`multiplier<doxid-structgls__control_1ac8bfb1ed777319ef92b7039c66f9a9b0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce<doxid-structgls__control_1a595df8d359282d27f49ac529283c509a>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`u<doxid-structgls__control_1abb669b70ee8fb00689add7fad23ce00f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`switch_full<doxid-structgls__control_1a425fe61f3ab00924d3f1e05e93985df5>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`drop<doxid-structgls__control_1aa340d2fee9ccb7c4a2f38ac086482506>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`tolerance<doxid-structgls__control_1ad0dcb73e98bad740852a54d6b7d1f6c2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cgce<doxid-structgls__control_1ada0fc18d7059672071369decb96f6178>`;
		bool :ref:`diagonal_pivoting<doxid-structgls__control_1aae1a90daac2378be15e6d9d644d19281>`;
		bool :ref:`struct_abort<doxid-structgls__control_1a9b775b57e2c00f53ca8d8a33e5589533>`;
	};
.. _details-structgls__control:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structgls__control_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; lp
.. _doxid-structgls__control_1a3eec33a4e6d8295c25d117bb25dd1b9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lp

Unit for error messages.

.. index:: pair: variable; wp
.. _doxid-structgls__control_1af203f5ddbac4a47afed1a07b97e7e477:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` wp

Unit for warning messages.

.. index:: pair: variable; mp
.. _doxid-structgls__control_1a4b5efeeced2b749191f71afc3bc8bebd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` mp

Unit for monitor output.

.. index:: pair: variable; ldiag
.. _doxid-structgls__control_1ad0e905fe282a7125424a53219afc0791:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ldiag

Controls level of diagnostic output.

.. index:: pair: variable; btf
.. _doxid-structgls__control_1a48e951043b7ec1aab5e2e1b62f1d4021:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` btf

Minimum block size for block-triangular form (BTF). Set to $n$ to avoid.

.. index:: pair: variable; maxit
.. _doxid-structgls__control_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

Maximum number of iterations.

.. index:: pair: variable; factor_blocking
.. _doxid-structgls__control_1abca1e309e7d2d73481534b8fdad872e7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factor_blocking

Level 3 blocking in factorize.

.. index:: pair: variable; solve_blas
.. _doxid-structgls__control_1ab2cc1341183896b96cacc4c6a39acdde:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` solve_blas

Switch for using Level 1 or 2 BLAS in solve.

.. index:: pair: variable; la
.. _doxid-structgls__control_1a8e48dcc59e4b8bbe40fe5b58321e4e72:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la

Initial size for real array for the factors.

.. index:: pair: variable; la_int
.. _doxid-structgls__control_1a7d8b6a8163160b2aa8f730ead5dd3727:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` la_int

Initial size for integer array for the factors.

.. index:: pair: variable; maxla
.. _doxid-structgls__control_1a6437ead17fd48daf197640949e8d4ff3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxla

Maximum size for real array for the factors.

.. index:: pair: variable; pivoting
.. _doxid-structgls__control_1aa4d2d93f87da5df80b8aa2bce688c030:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` pivoting

Controls pivoting: Number of columns searched. Zero for Markowitz.

.. index:: pair: variable; fill_in
.. _doxid-structgls__control_1afdaf7f9c41586b488e08ac26dd5d93b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` fill_in

Initially fill_in \* ne space allocated for factors.

.. index:: pair: variable; multiplier
.. _doxid-structgls__control_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` multiplier

Factor by which arrays sizes are to be increased if they are too small.

.. index:: pair: variable; reduce
.. _doxid-structgls__control_1a595df8d359282d27f49ac529283c509a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce

if previously allocated internal workspace arrays are greater than reduce times the currently required sizes, they are reset to current requirment

.. index:: pair: variable; u
.. _doxid-structgls__control_1abb669b70ee8fb00689add7fad23ce00f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` u

Pivot threshold.

.. index:: pair: variable; switch_full
.. _doxid-structgls__control_1a425fe61f3ab00924d3f1e05e93985df5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` switch_full

Density for switch to full code.

.. index:: pair: variable; drop
.. _doxid-structgls__control_1aa340d2fee9ccb7c4a2f38ac086482506:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` drop

Drop tolerance.

.. index:: pair: variable; tolerance
.. _doxid-structgls__control_1ad0dcb73e98bad740852a54d6b7d1f6c2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` tolerance

anything < this is considered zero

.. index:: pair: variable; cgce
.. _doxid-structgls__control_1ada0fc18d7059672071369decb96f6178:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cgce

Ratio for required reduction using IR.

.. index:: pair: variable; diagonal_pivoting
.. _doxid-structgls__control_1aae1a90daac2378be15e6d9d644d19281:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool diagonal_pivoting

Set to 0 for diagonal pivoting.

.. index:: pair: variable; struct_abort
.. _doxid-structgls__control_1a9b775b57e2c00f53ca8d8a33e5589533:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool struct_abort

Control to abort if structurally singular.

