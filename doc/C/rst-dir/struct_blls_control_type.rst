.. index:: pair: struct; blls_control_type
.. _doxid-structblls__control__type:

blls_control_type structure
---------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_blls.h>
	
	struct blls_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structblls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structblls__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structblls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structblls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structblls__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structblls__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structblls__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structblls__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cold_start<doxid-structblls__control__type_1ad5e3138a19f7400e9d5c8105fa724831>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`preconditioner<doxid-structblls__control__type_1adf7719f1a4491459e361e80a00c55656>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ratio_cg_vs_sd<doxid-structblls__control__type_1ab589a429c71e34b9c07c4d79a1e02902>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`change_max<doxid-structblls__control__type_1a4a70200c62828c4d82e2e3efa5ebdac4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structblls__control__type_1a7a1029142a22f3e2a1963c3428276849>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`arcsearch_max_steps<doxid-structblls__control__type_1a7dceaf3624973df51f1baa0937420517>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structblls__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`weight<doxid-structblls__control__type_1adcd20aeaf7042e972ddab56f3867ce70>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structblls__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_d<doxid-structblls__control__type_1a12784541c48f57127781bc1c5937c616>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structblls__control__type_1abc74ac9bbf6375075f8943aac6ee09e4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_relative<doxid-structblls__control__type_1acd5b41623ff5db9a81dc5a8421fe5e2f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :target:`stop_cg_absolute<doxid-structblls__control__type_1ad8ba10f33e624074c203f079afed54f8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_max<doxid-structblls__control__type_1a283df1a15f52c266a119dc0b37262f93>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_initial<doxid-structblls__control__type_1a9c84d5b6142e91ff0c56960bcdcf9eb5>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_reduction<doxid-structblls__control__type_1a94762277f055ecfb1c48bd439066fb21>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`arcsearch_acceptance_tol<doxid-structblls__control__type_1ac73142611027bffbf7cdb6552704152b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stabilisation_weight<doxid-structblls__control__type_1a6024e0b85e16b28075c69743433267cd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structblls__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		bool :ref:`direct_subproblem_solve<doxid-structblls__control__type_1ade9a921baa01ea4f50c39b7f640cd416>`;
		bool :ref:`exact_arc_search<doxid-structblls__control__type_1abd8ff06ab50d29e781cf407702346a4b>`;
		bool :ref:`advance<doxid-structblls__control__type_1adc86e2518dfd985256913c727dd2b84a>`;
		bool :ref:`space_critical<doxid-structblls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structblls__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`generate_sif_file<doxid-structblls__control__type_1aa75b3a16d146c0d7ad57bf9817033843>`;
		char :ref:`sif_file_name<doxid-structblls__control__type_1aaa95e830b709da79d9790471bab54193>`[31];
		char :ref:`prefix<doxid-structblls__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structblls__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a>`;
		struct :ref:`convert_control_type<doxid-structconvert__control__type>` :ref:`convert_control<doxid-structblls__control__type_1ad5c7fdb1978f9943554af877e3e1e37b>`;
	};
.. _details-structblls__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structblls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structblls__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit number for error and warning diagnostics

.. index:: pair: variable; out
.. _doxid-structblls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output unit number

.. index:: pair: variable; print_level
.. _doxid-structblls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required

.. index:: pair: variable; start_print
.. _doxid-structblls__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

on which iteration to start printing

.. index:: pair: variable; stop_print
.. _doxid-structblls__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

on which iteration to stop printing

.. index:: pair: variable; print_gap
.. _doxid-structblls__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

how many iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structblls__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

how many iterations to perform (-ve reverts to HUGE(1)-1)

.. index:: pair: variable; cold_start
.. _doxid-structblls__control__type_1ad5e3138a19f7400e9d5c8105fa724831:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cold_start

cold_start should be set to 0 if a warm start is required (with variable assigned according to X_stat, see below), and to any other value if the values given in prob.X suffice

.. index:: pair: variable; preconditioner
.. _doxid-structblls__control__type_1adf7719f1a4491459e361e80a00c55656:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` preconditioner

the preconditioner (scaling) used. Possible values are: /li 0. no preconditioner. /li 1. a diagonal preconditioner that normalizes the rows of $A$. /li anything else. a preconditioner supplied by the user either via a subroutine call of eval_prec} or via reverse communication.

.. index:: pair: variable; ratio_cg_vs_sd
.. _doxid-structblls__control__type_1ab589a429c71e34b9c07c4d79a1e02902:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ratio_cg_vs_sd

the ratio of how many iterations use CGLS rather than steepest descent

.. index:: pair: variable; change_max
.. _doxid-structblls__control__type_1a4a70200c62828c4d82e2e3efa5ebdac4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` change_max

the maximum number of per-iteration changes in the working set permitted when allowing CGLS rather than steepest descent

.. index:: pair: variable; cg_maxit
.. _doxid-structblls__control__type_1a7a1029142a22f3e2a1963c3428276849:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

how many CG iterations to perform per BLLS iteration (-ve reverts to n+1)

.. index:: pair: variable; arcsearch_max_steps
.. _doxid-structblls__control__type_1a7dceaf3624973df51f1baa0937420517:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` arcsearch_max_steps

the maximum number of steps allowed in a piecewise arcsearch (-ve=infini

.. index:: pair: variable; sif_file_device
.. _doxid-structblls__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

the unit number to write generated SIF file describing the current probl

.. index:: pair: variable; weight
.. _doxid-structblls__control__type_1adcd20aeaf7042e972ddab56f3867ce70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` weight

the value of the non-negative regularization weight sigma, i.e., the quadratic objective function q(x) will be regularized by adding 1/2 weight \|\|x\|\|^2; any value smaller than zero will be regarded as zero.

.. index:: pair: variable; infinity
.. _doxid-structblls__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_d
.. _doxid-structblls__control__type_1a12784541c48f57127781bc1c5937c616:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structblls__control__type_1abc74ac9bbf6375075f8943aac6ee09e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; stop_cg_relative
.. _doxid-structblls__control__type_1acd5b41623ff5db9a81dc5a8421fe5e2f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute)

.. index:: pair: variable; alpha_max
.. _doxid-structblls__control__type_1a283df1a15f52c266a119dc0b37262f93:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_max

the largest permitted arc length during the piecewise line search

.. index:: pair: variable; alpha_initial
.. _doxid-structblls__control__type_1a9c84d5b6142e91ff0c56960bcdcf9eb5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_initial

the initial arc length during the inexact piecewise line search

.. index:: pair: variable; alpha_reduction
.. _doxid-structblls__control__type_1a94762277f055ecfb1c48bd439066fb21:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_reduction

the arc length reduction factor for the inexact piecewise line search

.. index:: pair: variable; arcsearch_acceptance_tol
.. _doxid-structblls__control__type_1ac73142611027bffbf7cdb6552704152b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` arcsearch_acceptance_tol

the required relative reduction during the inexact piecewise line search

.. index:: pair: variable; stabilisation_weight
.. _doxid-structblls__control__type_1a6024e0b85e16b28075c69743433267cd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stabilisation_weight

the stabilisation weight added to the search-direction subproblem

.. index:: pair: variable; cpu_time_limit
.. _doxid-structblls__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve = no limit)

.. index:: pair: variable; direct_subproblem_solve
.. _doxid-structblls__control__type_1ade9a921baa01ea4f50c39b7f640cd416:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool direct_subproblem_solve

direct_subproblem_solve is true if the least-squares subproblem is to be solved using a matrix factorization, and false if conjugate gradients are to be preferred

.. index:: pair: variable; exact_arc_search
.. _doxid-structblls__control__type_1abd8ff06ab50d29e781cf407702346a4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_arc_search

exact_arc_search is true if an exact arc_search is required, and false if an approximation suffices

.. index:: pair: variable; advance
.. _doxid-structblls__control__type_1adc86e2518dfd985256913c727dd2b84a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool advance

advance is true if an inexact exact arc_search can increase steps as well as decrease them

.. index:: pair: variable; space_critical
.. _doxid-structblls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation times

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structblls__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structblls__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if generate_sif_file is true, a SIF file describing the current problem will be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structblls__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name (max 30 characters) of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structblls__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by a string (max 30 characters) prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structblls__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; convert_control
.. _doxid-structblls__control__type_1ad5c7fdb1978f9943554af877e3e1e37b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`convert_control_type<doxid-structconvert__control__type>` convert_control

control parameters for CONVERT

