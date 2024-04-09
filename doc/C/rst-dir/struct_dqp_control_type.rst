.. index:: pair: struct; dqp_control_type
.. _doxid-structdqp__control__type:

dqp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_dqp.h>
	
	struct dqp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structdqp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structdqp__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structdqp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structdqp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structdqp__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structdqp__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structdqp__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dual_starting_point<doxid-structdqp__control__type_1a416b1d999e77575f6f0cb5f150829ee4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structdqp__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_sc<doxid-structdqp__control__type_1a58895c9f8fe04a04578073223b30af2a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cauchy_only<doxid-structdqp__control__type_1afba2adad4ad503320066449f07aba83b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`arc_search_maxit<doxid-structdqp__control__type_1af633cdb2051e99e783dfe792a50a4149>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`cg_maxit<doxid-structdqp__control__type_1a7a1029142a22f3e2a1963c3428276849>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`explore_optimal_subspace<doxid-structdqp__control__type_1a166a5815eb8a029edd1b46dfc2854679>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`restore_problem<doxid-structdqp__control__type_1a19f10912888ac7ebd153fb21baaeaefa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sif_file_device<doxid-structdqp__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`qplib_file_device<doxid-structdqp__control__type_1a580c343e54a25a2d687782410c9b6917>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`rho<doxid-structdqp__control__type_1a7a571cc854bea7cc6488175668c15b3d>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structdqp__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_p<doxid-structdqp__control__type_1a3749f9dcb2eeb60815a18c85a7d7d440>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_p<doxid-structdqp__control__type_1a068065b9d50d5b222fbc8710d530bd9e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_d<doxid-structdqp__control__type_1a67b6a7be5dbfa34e9db4c960943fe31f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_d<doxid-structdqp__control__type_1ab479c27e2044c310e8d0c86869ea2307>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_abs_c<doxid-structdqp__control__type_1a1ed92b5ffb35957c5a8a0e657e312820>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_rel_c<doxid-structdqp__control__type_1a5c131e3b6061c09670e9c1959b6585a3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_relative<doxid-structdqp__control__type_1acd5b41623ff5db9a81dc5a8421fe5e2f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_cg_absolute<doxid-structdqp__control__type_1ad8ba10f33e624074c203f079afed54f8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cg_zero_curvature<doxid-structdqp__control__type_1aa72b520b0841cce9015e03522f58103b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_growth<doxid-structdqp__control__type_1a2c39ca5cec2100192d5b451678e61864>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structdqp__control__type_1abc74ac9bbf6375075f8943aac6ee09e4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structdqp__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structdqp__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`initial_perturbation<doxid-structdqp__control__type_1a6ef6c7936fe36bc165ba087eca79d9e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`perturbation_reduction<doxid-structdqp__control__type_1a12423a6a8d2151baede20265bee496ad>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`final_perturbation<doxid-structdqp__control__type_1a2c77f53cca5940c027611b0c5c9531c7>`;
		bool :ref:`factor_optimal_matrix<doxid-structdqp__control__type_1a64ad50a4e86e202661afcd04f340a2c8>`;
		bool :ref:`remove_dependencies<doxid-structdqp__control__type_1ae17a6b550239434c639239ddf45bc1ad>`;
		bool :ref:`treat_zero_bounds_as_general<doxid-structdqp__control__type_1a59b23877e9c8cb49f1f6261e24660295>`;
		bool :ref:`exact_arc_search<doxid-structdqp__control__type_1abd8ff06ab50d29e781cf407702346a4b>`;
		bool :ref:`subspace_direct<doxid-structdqp__control__type_1aa2b787b6188444e3c8f06fd4af4981a9>`;
		bool :ref:`subspace_alternate<doxid-structdqp__control__type_1ac45bab9097874bdfb86ce77bc19a0003>`;
		bool :ref:`subspace_arc_search<doxid-structdqp__control__type_1a166d14e55068610829e14b1616fa179a>`;
		bool :ref:`space_critical<doxid-structdqp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structdqp__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`generate_sif_file<doxid-structdqp__control__type_1aa75b3a16d146c0d7ad57bf9817033843>`;
		bool :ref:`generate_qplib_file<doxid-structdqp__control__type_1ad226b26090e539cf47228ba5ec7dc08d>`;
		char :ref:`symmetric_linear_solver<doxid-structdqp__control__type_1af297ace351b9307640715643cde57384>`[31];
		char :ref:`definite_linear_solver<doxid-structdqp__control__type_1a9b46b7a8e0af020499e645bef711f634>`[31];
		char :ref:`unsymmetric_linear_solver<doxid-structdqp__control__type_1aef6da6b715a0f41983c2a62397104eec>`[31];
		char :ref:`sif_file_name<doxid-structdqp__control__type_1aaa95e830b709da79d9790471bab54193>`[31];
		char :ref:`qplib_file_name<doxid-structdqp__control__type_1a3d36354e9f30d642f8b081ba85d777d3>`[31];
		char :ref:`prefix<doxid-structdqp__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>` :ref:`fdc_control<doxid-structdqp__control__type_1a7bef6e4f678e16a4dcdc40677efddd80>`;
		struct :ref:`sls_control_type<doxid-structsls__control__type>` :ref:`sls_control<doxid-structdqp__control__type_1a31b308b91955ee385daacc3de00f161b>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structdqp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a>`;
		struct :ref:`gltr_control_type<doxid-structgltr__control__type>` :ref:`gltr_control<doxid-structdqp__control__type_1aa48d482633f3788830b1d8dc85fa91d6>`;
	};
.. _details-structdqp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structdqp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structdqp__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structdqp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structdqp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structdqp__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structdqp__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structdqp__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

printing will only occur every print_gap iterations

.. index:: pair: variable; dual_starting_point
.. _doxid-structdqp__control__type_1a416b1d999e77575f6f0cb5f150829ee4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dual_starting_point

which starting point should be used for the dual problem

* -1 user supplied comparing primal vs dual variables

* 0 user supplied

* 1 minimize linearized dual

* 2 minimize simplified quadratic dual

* 3 all free (= all active primal costraints)

* 4 all fixed on bounds (= no active primal costraints)

.. index:: pair: variable; maxit
.. _doxid-structdqp__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; max_sc
.. _doxid-structdqp__control__type_1a58895c9f8fe04a04578073223b30af2a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_sc

the maximum permitted size of the Schur complement before a refactorization is performed (used in the case where there is no Fredholm Alternative, 0 = refactor every iteration)

.. index:: pair: variable; cauchy_only
.. _doxid-structdqp__control__type_1afba2adad4ad503320066449f07aba83b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cauchy_only

a subspace step will only be taken when the current Cauchy step has changed no more than than cauchy_only active constraints; the subspace step will always be taken if cauchy_only < 0

.. index:: pair: variable; arc_search_maxit
.. _doxid-structdqp__control__type_1af633cdb2051e99e783dfe792a50a4149:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` arc_search_maxit

how many iterations are allowed per arc search (-ve = as many as require

.. index:: pair: variable; cg_maxit
.. _doxid-structdqp__control__type_1a7a1029142a22f3e2a1963c3428276849:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` cg_maxit

how many CG iterations to perform per DQP iteration (-ve reverts to n+1)

.. index:: pair: variable; explore_optimal_subspace
.. _doxid-structdqp__control__type_1a166a5815eb8a029edd1b46dfc2854679:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` explore_optimal_subspace

once a potentially optimal subspace has been found, investigate it

* 0 as per an ordinary subspace

* 1 by increasing the maximum number of allowed CG iterations

* 2 by switching to a direct method

.. index:: pair: variable; restore_problem
.. _doxid-structdqp__control__type_1a19f10912888ac7ebd153fb21baaeaefa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; sif_file_device
.. _doxid-structdqp__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; qplib_file_device
.. _doxid-structdqp__control__type_1a580c343e54a25a2d687782410c9b6917:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` qplib_file_device

specifies the unit number to write generated QPLIB file describing the current problem

.. index:: pair: variable; rho
.. _doxid-structdqp__control__type_1a7a571cc854bea7cc6488175668c15b3d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` rho

the penalty weight, rho. The general constraints are not enforced explicitly, but instead included in the objective as a penalty term weighted by rho when rho > 0. If rho <= 0, the general constraints are explicit (that is, there is no penalty term in the objective function)

.. index:: pair: variable; infinity
.. _doxid-structdqp__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_abs_p
.. _doxid-structdqp__control__type_1a3749f9dcb2eeb60815a18c85a7d7d440:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_p

the required absolute and relative accuracies for the primal infeasibilies

.. index:: pair: variable; stop_rel_p
.. _doxid-structdqp__control__type_1a068065b9d50d5b222fbc8710d530bd9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_p

see stop_abs_p

.. index:: pair: variable; stop_abs_d
.. _doxid-structdqp__control__type_1a67b6a7be5dbfa34e9db4c960943fe31f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_d

the required absolute and relative accuracies for the dual infeasibility

.. index:: pair: variable; stop_rel_d
.. _doxid-structdqp__control__type_1ab479c27e2044c310e8d0c86869ea2307:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_d

see stop_abs_d

.. index:: pair: variable; stop_abs_c
.. _doxid-structdqp__control__type_1a1ed92b5ffb35957c5a8a0e657e312820:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_abs_c

the required absolute and relative accuracies for the complementarity

.. index:: pair: variable; stop_rel_c
.. _doxid-structdqp__control__type_1a5c131e3b6061c09670e9c1959b6585a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_rel_c

see stop_abs_c

.. index:: pair: variable; stop_cg_relative
.. _doxid-structdqp__control__type_1acd5b41623ff5db9a81dc5a8421fe5e2f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_relative

the CG iteration will be stopped as soon as the current norm of the preconditioned gradient is smaller than max( stop_cg_relative \* initial preconditioned gradient, stop_cg_absolute )

.. index:: pair: variable; stop_cg_absolute
.. _doxid-structdqp__control__type_1ad8ba10f33e624074c203f079afed54f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_cg_absolute

see stop_cg_relative

.. index:: pair: variable; cg_zero_curvature
.. _doxid-structdqp__control__type_1aa72b520b0841cce9015e03522f58103b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cg_zero_curvature

threshold below which curvature is regarded as zero if CG is used

.. index:: pair: variable; max_growth
.. _doxid-structdqp__control__type_1a2c39ca5cec2100192d5b451678e61864:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` max_growth

maximum growth factor allowed without a refactorization

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structdqp__control__type_1abc74ac9bbf6375075f8943aac6ee09e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; cpu_time_limit
.. _doxid-structdqp__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structdqp__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; initial_perturbation
.. _doxid-structdqp__control__type_1a6ef6c7936fe36bc165ba087eca79d9e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` initial_perturbation

the initial penalty weight (for DLP only)

.. index:: pair: variable; perturbation_reduction
.. _doxid-structdqp__control__type_1a12423a6a8d2151baede20265bee496ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` perturbation_reduction

the penalty weight reduction factor (for DLP only)

.. index:: pair: variable; final_perturbation
.. _doxid-structdqp__control__type_1a2c77f53cca5940c027611b0c5c9531c7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` final_perturbation

the final penalty weight (for DLP only)

.. index:: pair: variable; factor_optimal_matrix
.. _doxid-structdqp__control__type_1a64ad50a4e86e202661afcd04f340a2c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool factor_optimal_matrix

are the factors of the optimal augmented matrix required? (for DLP only)

.. index:: pair: variable; remove_dependencies
.. _doxid-structdqp__control__type_1ae17a6b550239434c639239ddf45bc1ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structdqp__control__type_1a59b23877e9c8cb49f1f6261e24660295:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; exact_arc_search
.. _doxid-structdqp__control__type_1abd8ff06ab50d29e781cf407702346a4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool exact_arc_search

if .exact_arc_search is true, an exact piecewise arc search will be performed. Otherwise an ineaxt search using a backtracing Armijo strategy will be employed

.. index:: pair: variable; subspace_direct
.. _doxid-structdqp__control__type_1aa2b787b6188444e3c8f06fd4af4981a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subspace_direct

if .subspace_direct is true, the subspace step will be calculated using a direct (factorization) method, while if it is false, an iterative (conjugate-gradient) method will be used.

.. index:: pair: variable; subspace_alternate
.. _doxid-structdqp__control__type_1ac45bab9097874bdfb86ce77bc19a0003:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subspace_alternate

if .subspace_alternate is true, the subspace step will alternate between a direct (factorization) method and an iterative (GLTR conjugate-gradient) method. This will override .subspace_direct

.. index:: pair: variable; subspace_arc_search
.. _doxid-structdqp__control__type_1a166d14e55068610829e14b1616fa179a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool subspace_arc_search

if .subspace_arc_search is true, a piecewise arc search will be performed along the subspace step. Otherwise the search will stop at the firstconstraint encountered

.. index:: pair: variable; space_critical
.. _doxid-structdqp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structdqp__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structdqp__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; generate_qplib_file
.. _doxid-structdqp__control__type_1ad226b26090e539cf47228ba5ec7dc08d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool generate_qplib_file

if .generate_qplib_file is .true. if a QPLIB file describing the current problem is to be generated

.. index:: pair: variable; symmetric_linear_solver
.. _doxid-structdqp__control__type_1af297ace351b9307640715643cde57384:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char symmetric_linear_solver[31]

the name of the symmetric-indefinite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', and 'sytr', although only 'sytr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; definite_linear_solver
.. _doxid-structdqp__control__type_1a9b46b7a8e0af020499e645bef711f634:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char definite_linear_solver[31]

the name of the definite linear equation solver used. Possible choices are currently: 'sils', 'ma27', 'ma57', 'ma77', 'ma86', 'ma87', 'ma97', 'ssids', 'mumps', 'pardiso', 'mkl_pardiso', 'pastix', 'wsmp', 'potr',  'sytr' and 'pbtr', although only 'potr',  'sytr', 'pbtr' and, for OMP 4.0-compliant compilers, 'ssids' are installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_sls<details-sls__solvers>`.

.. index:: pair: variable; unsymmetric_linear_solver
.. _doxid-structdqp__control__type_1aef6da6b715a0f41983c2a62397104eec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char unsymmetric_linear_solver[31]

the name of the unsymmetric linear equation solver used. Possible choices are currently: 'gls', 'ma48' and 'getr', although only 'getr' is installed by default; others are easily installed (see README.external). More details of the capabilities of each solver are provided in the documentation for :ref:`galahad_uls<details-uls__solvers>`.

.. index:: pair: variable; sif_file_name
.. _doxid-structdqp__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char sif_file_name[31]

name of generated SIF file containing input problem

.. index:: pair: variable; qplib_file_name
.. _doxid-structdqp__control__type_1a3d36354e9f30d642f8b081ba85d777d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char qplib_file_name[31]

name of generated QPLIB file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structdqp__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structdqp__control__type_1a7bef6e4f678e16a4dcdc40677efddd80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sls_control
.. _doxid-structdqp__control__type_1a31b308b91955ee385daacc3de00f161b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; sbls_control
.. _doxid-structdqp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

.. index:: pair: variable; gltr_control
.. _doxid-structdqp__control__type_1aa48d482633f3788830b1d8dc85fa91d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`gltr_control_type<doxid-structgltr__control__type>` gltr_control

control parameters for GLTR

