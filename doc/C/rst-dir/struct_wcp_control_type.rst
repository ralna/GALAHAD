.. index:: pair: struct; wcp_control_type
.. _doxid-structwcp__control__type:

wcp_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_wcp.h>
	
	struct wcp_control_type {
		// components
	
		bool :ref:`f_indexing<doxid-structwcp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structwcp__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structwcp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structwcp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structwcp__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structwcp__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structwcp__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`initial_point<doxid-structwcp__control__type_1a4b0d426781b9f3ffabcb4c55cbe4e01d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`factor<doxid-structwcp__control__type_1a42eac05293c31fef9b90c92698928d7d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_col<doxid-structwcp__control__type_1abca2db33b9520095e98790d45a1be93f>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`indmin<doxid-structwcp__control__type_1a5031bbc31f94e4cba6a540a3182b6d80>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`valmin<doxid-structwcp__control__type_1a0e142fa8dc9c363c3c2993b6129b0955>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`itref_max<doxid-structwcp__control__type_1a903ba4ef0869186a65d4c32459a6a0ed>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`infeas_max<doxid-structwcp__control__type_1af7d33b88b67b2366d7e2df31d42083a9>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`perturbation_strategy<doxid-structwcp__control__type_1a890441c8643aa00c556c54171f2e5ae6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`restore_problem<doxid-structwcp__control__type_1a19f10912888ac7ebd153fb21baaeaefa>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structwcp__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_p<doxid-structwcp__control__type_1a8933604acacc0fb4367caac730b6c79b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_d<doxid-structwcp__control__type_1a12784541c48f57127781bc1c5937c616>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_c<doxid-structwcp__control__type_1aec5ea9177505eb7723a8e092535556cb>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`prfeas<doxid-structwcp__control__type_1a09a0a5771a0300078ebe5f344ef4e492>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`dufeas<doxid-structwcp__control__type_1a2ac34c5838499ed4992037655f52134a>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_target<doxid-structwcp__control__type_1afaa9812f90e3ee012acb65f1553a13ae>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_accept_fraction<doxid-structwcp__control__type_1a7219b2e1551923beca47c098f37731f3>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`mu_increase_factor<doxid-structwcp__control__type_1a7379643e1a11c2df50758fd4b0604866>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`required_infeas_reduction<doxid-structwcp__control__type_1afea4bb58c0b0e93bdadf50dd76609f2b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`implicit_tol<doxid-structwcp__control__type_1ad1382b780e6eca1515eba22e386ff636>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structwcp__control__type_1a133347eb5f45a24a77b63b4afd4212e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol_for_dependencies<doxid-structwcp__control__type_1a14e253b53c59b0850d7d3b5245d89df9>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`zero_pivot<doxid-structwcp__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`perturb_start<doxid-structwcp__control__type_1aa1f8e046febf2491671fbe5701dde416>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`alpha_scale<doxid-structwcp__control__type_1a81c6e54fe4cd62154ac77c8d74175472>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`identical_bounds_tol<doxid-structwcp__control__type_1abc74ac9bbf6375075f8943aac6ee09e4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_perturb_factor<doxid-structwcp__control__type_1a203c6477914c5ff2df8d1d34dc730cf4>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reduce_perturb_multiplier<doxid-structwcp__control__type_1abd3c57a6340705f4bb0e63af8e0aad81>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`insufficiently_feasible<doxid-structwcp__control__type_1a2c44c2b6e8ef45336e4418a7f1850021>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`perturbation_small<doxid-structwcp__control__type_1ab66799758f41fd5be1f0701e3264f8de>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structwcp__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structwcp__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`remove_dependencies<doxid-structwcp__control__type_1ae17a6b550239434c639239ddf45bc1ad>`;
		bool :ref:`treat_zero_bounds_as_general<doxid-structwcp__control__type_1a59b23877e9c8cb49f1f6261e24660295>`;
		bool :ref:`just_feasible<doxid-structwcp__control__type_1a1337f1d22b070690c78f25f7ecaf1e96>`;
		bool :ref:`balance_initial_complementarity<doxid-structwcp__control__type_1a0dc49ffb06b68e53b4efe7f609c97439>`;
		bool :ref:`use_corrector<doxid-structwcp__control__type_1a92a646996c4c1099e32ddf75f7c7e976>`;
		bool :ref:`space_critical<doxid-structwcp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structwcp__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		bool :ref:`record_x_status<doxid-structwcp__control__type_1a369ea250a3eff48788e476bd6e8e5b74>`;
		bool :ref:`record_c_status<doxid-structwcp__control__type_1af40052ed2e177e61a290faaf618cb282>`;
		char :ref:`prefix<doxid-structwcp__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
		struct :ref:`fdc_control_type<doxid-structfdc__control__type>` :ref:`fdc_control<doxid-structwcp__control__type_1a7bef6e4f678e16a4dcdc40677efddd80>`;
		struct :ref:`sbls_control_type<doxid-structsbls__control__type>` :ref:`sbls_control<doxid-structwcp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a>`;
	};
.. _details-structwcp__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structwcp__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structwcp__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structwcp__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structwcp__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required is specified by print_level

.. index:: pair: variable; start_print
.. _doxid-structwcp__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structwcp__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structwcp__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; initial_point
.. _doxid-structwcp__control__type_1a4b0d426781b9f3ffabcb4c55cbe4e01d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` initial_point

how to choose the initial point. Possible values are

* 0 the values input in X, shifted to be at least prfeas from their nearest bound, will be used

* 1 the nearest point to the "bound average" 0.5(X_l+X_u) that satisfies the linear constraints will be used

.. index:: pair: variable; factor
.. _doxid-structwcp__control__type_1a42eac05293c31fef9b90c92698928d7d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` factor

the factorization to be used. Possible values are

* 0 automatic

* 1 Schur-complement factorization

* 2 augmented-system factorization

.. index:: pair: variable; max_col
.. _doxid-structwcp__control__type_1abca2db33b9520095e98790d45a1be93f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_col

the maximum number of nonzeros in a column of A which is permitted with the Schur-complement factorization

.. index:: pair: variable; indmin
.. _doxid-structwcp__control__type_1a5031bbc31f94e4cba6a540a3182b6d80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` indmin

an initial guess as to the integer workspace required by SBLS

.. index:: pair: variable; valmin
.. _doxid-structwcp__control__type_1a0e142fa8dc9c363c3c2993b6129b0955:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` valmin

an initial guess as to the real workspace required by SBLS

.. index:: pair: variable; itref_max
.. _doxid-structwcp__control__type_1a903ba4ef0869186a65d4c32459a6a0ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` itref_max

the maximum number of iterative refinements allowed

.. index:: pair: variable; infeas_max
.. _doxid-structwcp__control__type_1af7d33b88b67b2366d7e2df31d42083a9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` infeas_max

the number of iterations for which the overall infeasibility of the problem is not reduced by at least a factor .required_infeas_reduction before the problem is flagged as infeasible (see required_infeas_reducti

.. index:: pair: variable; perturbation_strategy
.. _doxid-structwcp__control__type_1a890441c8643aa00c556c54171f2e5ae6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` perturbation_strategy

the strategy used to reduce relaxed constraint bounds. Possible values are

* 0 do not perturb the constraints

* 1 reduce all perturbations by the same amount with linear reduction

* 2 reduce each perturbation as much as possible with linear reduction

* 3 reduce all perturbations by the same amount with superlinear reduction

* 4 reduce each perturbation as much as possible with superlinear reduction

.. index:: pair: variable; restore_problem
.. _doxid-structwcp__control__type_1a19f10912888ac7ebd153fb21baaeaefa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; infinity
.. _doxid-structwcp__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_p
.. _doxid-structwcp__control__type_1a8933604acacc0fb4367caac730b6c79b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_p

the required accuracy for the primal infeasibility

.. index:: pair: variable; stop_d
.. _doxid-structwcp__control__type_1a12784541c48f57127781bc1c5937c616:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_d

the required accuracy for the dual infeasibility

.. index:: pair: variable; stop_c
.. _doxid-structwcp__control__type_1aec5ea9177505eb7723a8e092535556cb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_c

the required accuracy for the complementarity

.. index:: pair: variable; prfeas
.. _doxid-structwcp__control__type_1a09a0a5771a0300078ebe5f344ef4e492:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` prfeas

initial primal variables will not be closer than prfeas from their bound

.. index:: pair: variable; dufeas
.. _doxid-structwcp__control__type_1a2ac34c5838499ed4992037655f52134a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` dufeas

initial dual variables will not be closer than dufeas from their bounds

.. index:: pair: variable; mu_target
.. _doxid-structwcp__control__type_1afaa9812f90e3ee012acb65f1553a13ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_target

the target value of the barrier parameter. If mu_target is not positive, it will be reset to an appropriate value

.. index:: pair: variable; mu_accept_fraction
.. _doxid-structwcp__control__type_1a7219b2e1551923beca47c098f37731f3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_accept_fraction

the complemtary slackness x_i.z_i will be judged to lie within an acceptable margin around its target value mu as soon as mu_accept_fraction \* mu <= x_i.z_i <= ( 1 / mu_accept_fraction ) \* mu; the perturbations will be reduced as soon as all of the complemtary slacknesses x_i.z_i lie within acceptable bounds. mu_accept_fraction will be reset to ensure that it lies in the interval (0,1]

.. index:: pair: variable; mu_increase_factor
.. _doxid-structwcp__control__type_1a7379643e1a11c2df50758fd4b0604866:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` mu_increase_factor

the target value of the barrier parameter will be increased by mu_increase_factor for infeasible constraints every time the perturbations are adjusted

.. index:: pair: variable; required_infeas_reduction
.. _doxid-structwcp__control__type_1afea4bb58c0b0e93bdadf50dd76609f2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` required_infeas_reduction

if the overall infeasibility of the problem is not reduced by at least a factor required_infeas_reduction over .infeas_max iterations, the problem is flagged as infeasible (see infeas_max)

.. index:: pair: variable; implicit_tol
.. _doxid-structwcp__control__type_1ad1382b780e6eca1515eba22e386ff636:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` implicit_tol

any primal or dual variable that is less feasible than implicit_tol will be regarded as defining an implicit constraint

.. index:: pair: variable; pivot_tol
.. _doxid-structwcp__control__type_1a133347eb5f45a24a77b63b4afd4212e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

the threshold pivot used by the matrix factorization. See the documentation for SBLS for details (obsolete)

.. index:: pair: variable; pivot_tol_for_dependencies
.. _doxid-structwcp__control__type_1a14e253b53c59b0850d7d3b5245d89df9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol_for_dependencies

the threshold pivot used by the matrix factorization when attempting to detect linearly dependent constraints. See the documentation for SBLS for details (obsolete)

.. index:: pair: variable; zero_pivot
.. _doxid-structwcp__control__type_1aed8525bc028ed7ae0a9dd1bb3154cda2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` zero_pivot

any pivots smaller than zero_pivot in absolute value will be regarded to zero when attempting to detect linearly dependent constraints (obsolete)

.. index:: pair: variable; perturb_start
.. _doxid-structwcp__control__type_1aa1f8e046febf2491671fbe5701dde416:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` perturb_start

the constraint bounds will initially be relaxed by perturb_start; this perturbation will subsequently be reduced to zero. If perturb_start < 0, the amount by which the bounds are relaxed will be computed automatically

.. index:: pair: variable; alpha_scale
.. _doxid-structwcp__control__type_1a81c6e54fe4cd62154ac77c8d74175472:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` alpha_scale

the test for rank defficiency will be to factorize 
$$\left(\begin{array}{cc}\alpha I & A^T_{\cal E} \\ A^{}_{\cal E} & 0
\end{array}\right)$$
for $\alpha =$ alpha_scale

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structwcp__control__type_1abc74ac9bbf6375075f8943aac6ee09e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` identical_bounds_tol

any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer tha identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; reduce_perturb_factor
.. _doxid-structwcp__control__type_1a203c6477914c5ff2df8d1d34dc730cf4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_perturb_factor

the constraint perturbation will be reduced as follows:

* - if the variable lies outside a bound, the corresponding perturbation will be reduced to reduce_perturb_factor \* current pertubation
  
  * 
    * ( 1 - reduce_perturb_factor ) \* violation

* - otherwise, if the variable lies within insufficiently_feasible of its bound the pertubation will be reduced to 

  *
    * reduce_perturb_multiplier \* current pertubation

* - otherwise if will be set to zero

.. index:: pair: variable; reduce_perturb_multiplier
.. _doxid-structwcp__control__type_1abd3c57a6340705f4bb0e63af8e0aad81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reduce_perturb_multiplier

see reduce_perturb_factor

.. index:: pair: variable; insufficiently_feasible
.. _doxid-structwcp__control__type_1a2c44c2b6e8ef45336e4418a7f1850021:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` insufficiently_feasible

see reduce_perturb_factor

.. index:: pair: variable; perturbation_small
.. _doxid-structwcp__control__type_1ab66799758f41fd5be1f0701e3264f8de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` perturbation_small

if the maximum constraint pertubation is smaller than perturbation_small and the violation is smaller than implicit_tol, the method will deduce that there is a feasible point but no interior

.. index:: pair: variable; cpu_time_limit
.. _doxid-structwcp__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structwcp__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; remove_dependencies
.. _doxid-structwcp__control__type_1ae17a6b550239434c639239ddf45bc1ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structwcp__control__type_1a59b23877e9c8cb49f1f6261e24660295:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; just_feasible
.. _doxid-structwcp__control__type_1a1337f1d22b070690c78f25f7ecaf1e96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool just_feasible

if .just_feasible is true, the algorithm will stop as soon as a feasible point is found. Otherwise, the optimal solution to the problem will be found

.. index:: pair: variable; balance_initial_complementarity
.. _doxid-structwcp__control__type_1a0dc49ffb06b68e53b4efe7f609c97439:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool balance_initial_complementarity

if .balance_initial_complementarity is .true. the initial complemetarity will be balanced

.. index:: pair: variable; use_corrector
.. _doxid-structwcp__control__type_1a92a646996c4c1099e32ddf75f7c7e976:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool use_corrector

if .use_corrector, a corrector step will be used

.. index:: pair: variable; space_critical
.. _doxid-structwcp__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structwcp__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; record_x_status
.. _doxid-structwcp__control__type_1a369ea250a3eff48788e476bd6e8e5b74:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool record_x_status

if .record_x_status is true, the array inform.X_status will be allocated and the status of the bound constraints will be reported on exit.

.. index:: pair: variable; record_c_status
.. _doxid-structwcp__control__type_1af40052ed2e177e61a290faaf618cb282:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool record_c_status

if .record_c_status is true, the array inform.C_status will be allocated and the status of the general constraints will be reported on exit.

.. index:: pair: variable; prefix
.. _doxid-structwcp__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structwcp__control__type_1a7bef6e4f678e16a4dcdc40677efddd80:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sbls_control
.. _doxid-structwcp__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_control_type<doxid-structsbls__control__type>` sbls_control

control parameters for SBLS

