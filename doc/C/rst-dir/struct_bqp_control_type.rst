.. index:: pair: struct; bqp_control_type
.. _doxid-structbqp__control__type:

bqp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_bqp.h>
	
	struct bqp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structbqp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structbqp__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structbqp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structbqp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structbqp__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structbqp__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structbqp__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structbqp__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cold_start<doxid-structbqp__control__type_1ad5e3138a19f7400e9d5c8105fa724831>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`ratio_cg_vs_sd<doxid-structbqp__control__type_1ab589a429c71e34b9c07c4d79a1e02902>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`change_max<doxid-structbqp__control__type_1a4a70200c62828c4d82e2e3efa5ebdac4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structbqp__control__type_1a7a1029142a22f3e2a1963c3428276849>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structbqp__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structbqp__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_p<doxid-structbqp__control__type_1a8933604acacc0fb4367caac730b6c79b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_d<doxid-structbqp__control__type_1a12784541c48f57127781bc1c5937c616>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c<doxid-structbqp__control__type_1aec5ea9177505eb7723a8e092535556cb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structbqp__control__type_1abc74ac9bbf6375075f8943aac6ee09e4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_relative<doxid-structbqp__control__type_1acd5b41623ff5db9a81dc5a8421fe5e2f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_absolute<doxid-structbqp__control__type_1ad8ba10f33e624074c203f079afed54f8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_curvature<doxid-structbqp__control__type_1a148d5812162498605583e801b85da570>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structbqp__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		bool :ref:`exact_arcsearch<doxid-structbqp__control__type_1af81c1d274ed80b254924cd5c594203e6>`;
		bool :ref:`space_critical<doxid-structbqp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structbqp__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`generate_sif_file<doxid-structbqp__control__type_1aa75b3a16d146c0d7ad57bf9817033843>`;
		char :ref:`sif_file_name<doxid-structbqp__control__type_1aaa95e830b709da79d9790471bab54193>`[31];
		char :ref:`prefix<doxid-structbqp__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structbqp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a>`;
	};
.. _details-structbqp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbqp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbqp__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

unit number for error and warning diagnostics

.. index:: pair: variable; out
.. _doxid-structbqp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output unit number

.. index:: pair: variable; print_level
.. _doxid-structbqp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required

.. index:: pair: variable; start_print
.. _doxid-structbqp__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

on which iteration to start printing

.. index:: pair: variable; stop_print
.. _doxid-structbqp__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

on which iteration to stop printing

.. index:: pair: variable; print_gap
.. _doxid-structbqp__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

how many iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structbqp__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

how many iterations to perform (-ve reverts to HUGE(1)-1)

.. index:: pair: variable; cold_start
.. _doxid-structbqp__control__type_1ad5e3138a19f7400e9d5c8105fa724831:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cold_start

cold_start should be set to 0 if a warm start is required (with variable assigned according to B_stat, see below), and to any other value if the values given in prob.X suffice

.. index:: pair: variable; ratio_cg_vs_sd
.. _doxid-structbqp__control__type_1ab589a429c71e34b9c07c4d79a1e02902:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` ratio_cg_vs_sd

the ratio of how many iterations use CG rather steepest descent

.. index:: pair: variable; change_max
.. _doxid-structbqp__control__type_1a4a70200c62828c4d82e2e3efa5ebdac4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` change_max

the maximum number of per-iteration changes in the working set permitted when allowing CG rather than steepest descent

.. index:: pair: variable; cg_maxit
.. _doxid-structbqp__control__type_1a7a1029142a22f3e2a1963c3428276849:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

how many CG iterations to perform per BQP iteration (-ve reverts to n+1)

.. index:: pair: variable; sif_file_device
.. _doxid-structbqp__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structbqp__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_p
.. _doxid-structbqp__control__type_1a8933604acacc0fb4367caac730b6c79b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_p

the required accuracy for the primal infeasibility

.. index:: pair: variable; stop_d
.. _doxid-structbqp__control__type_1a12784541c48f57127781bc1c5937c616:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_c
.. _doxid-structbqp__control__type_1aec5ea9177505eb7723a8e092535556cb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c

the required accuracy for the complementary slackness

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structbqp__control__type_1abc74ac9bbf6375075f8943aac6ee09e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (x_l,x_u) that are closer than i dentical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; stop_cg_relative
.. _doxid-structbqp__control__type_1acd5b41623ff5db9a81dc5a8421fe5e2f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute)

.. index:: pair: variable; stop_cg_absolute
.. _doxid-structbqp__control__type_1ad8ba10f33e624074c203f079afed54f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_absolute

see stop_cg_relative

.. index:: pair: variable; zero_curvature
.. _doxid-structbqp__control__type_1a148d5812162498605583e801b85da570:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_curvature

threshold below which curvature is regarded as zero

.. index:: pair: variable; cpu_time_limit
.. _doxid-structbqp__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve = no limit)

.. index:: pair: variable; exact_arcsearch
.. _doxid-structbqp__control__type_1af81c1d274ed80b254924cd5c594203e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_arcsearch

exact_arcsearch is true if an exact arcsearch is required, and false if approximation suffices

.. index:: pair: variable; space_critical
.. _doxid-structbqp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation times

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbqp__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structbqp__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if generate_sif_file is true, a SIF file describing the current problem will be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structbqp__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name (max 30 characters) of generated SIF file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structbqp__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by a string (max 30 characters) prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; sbls_control
.. _doxid-structbqp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

