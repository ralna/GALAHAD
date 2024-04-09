.. index:: pair: struct; ugo_control_type
.. _doxid-structugo__control__type:

ugo_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_ugo.h>
	
	struct ugo_control_type {
		// components
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`error<doxid-structugo__control__type_1a11614f44ef4d939bdd984953346a7572>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structugo__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structugo__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`start_print<doxid-structugo__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`stop_print<doxid-structugo__control__type_1a9a3d9960a04602d2a18009c82ae2124e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_gap<doxid-structugo__control__type_1a31edaef6b722ef2721633484405a649b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`maxit<doxid-structugo__control__type_1ab717630b215f0362699acac11fb3652c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`initial_points<doxid-structugo__control__type_1a31cfe38db49ce764d93d56ea80a21bf5>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`storage_increment<doxid-structugo__control__type_1a1b1bbeca4053127c0829ac6e28505faf>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`buffer<doxid-structugo__control__type_1a0c7afc3fbe2c84da5f8852c76839b03b>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`lipschitz_estimate_used<doxid-structugo__control__type_1ab4b797b153fd1ac00231b032263276b1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`next_interval_selection<doxid-structugo__control__type_1a6a3ba03a92a96dfc5cd1c094e1e3bf87>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`refine_with_newton<doxid-structugo__control__type_1a6618fb3c3e71213bde65602fc84b92ac>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`alive_unit<doxid-structugo__control__type_1a3fc6359d77a53a63d57ea600b51eac13>`;
		char :ref:`alive_file<doxid-structugo__control__type_1ac631699a26f321b14dbed37115f3c006>`[31];
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`stop_length<doxid-structugo__control__type_1a6bf05a14c29051133abf2e66de24e460>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`small_g_for_newton<doxid-structugo__control__type_1af8026b35c403d956d877121aa8ec4e7c>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`small_g<doxid-structugo__control__type_1a7c2cf28f2ca5d11f2a013be0a3661e69>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`obj_sufficient<doxid-structugo__control__type_1a307b3c1c0f3796e8195b2339b6082b3b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`global_lipschitz_constant<doxid-structugo__control__type_1a1a37c2054aa6a285605bbfa6f0d5b73b>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`reliability_parameter<doxid-structugo__control__type_1a73f576b3c9911d11606f473feb090825>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`lipschitz_lower_bound<doxid-structugo__control__type_1aa114fdc06a2a81b1274d165448caa99e>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`cpu_time_limit<doxid-structugo__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`clock_time_limit<doxid-structugo__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8>`;
		bool :ref:`second_derivative_available<doxid-structugo__control__type_1ab91fc45050afb092b9f27ab2910b90d1>`;
		bool :ref:`space_critical<doxid-structugo__control__type_1a957fc1f4f26eeef3b0951791ff972e8d>`;
		bool :ref:`deallocate_error_fatal<doxid-structugo__control__type_1a58a2c67fad6e808e8365eff67700cba5>`;
		char :ref:`prefix<doxid-structugo__control__type_1a1dc05936393ba705f516a0c275df4ffc>`[31];
	};
.. _details-structugo__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structugo__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structugo__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structugo__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* 1 a one-line summary for every improvement

* 2 a summary of each iteration

* $\geq$ 3 increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structugo__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structugo__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structugo__control__type_1a31edaef6b722ef2721633484405a649b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structugo__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` maxit

the maximum number of iterations allowed

.. index:: pair: variable; initial_points
.. _doxid-structugo__control__type_1a31cfe38db49ce764d93d56ea80a21bf5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` initial_points

the number of initial (uniformly-spaced) evaluation points (<2 reset to 2)

.. index:: pair: variable; storage_increment
.. _doxid-structugo__control__type_1a1b1bbeca4053127c0829ac6e28505faf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` storage_increment

incremenets of storage allocated (less that 1000 will be reset to 1000)

.. index:: pair: variable; buffer
.. _doxid-structugo__control__type_1a0c7afc3fbe2c84da5f8852c76839b03b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` buffer

unit for any out-of-core writing when expanding arrays

.. index:: pair: variable; lipschitz_estimate_used
.. _doxid-structugo__control__type_1ab4b797b153fd1ac00231b032263276b1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` lipschitz_estimate_used

what sort of Lipschitz constant estimate will be used:

* 1 = global contant provided

* 2 = global contant estimated

* 3 = local costants estimated

.. index:: pair: variable; next_interval_selection
.. _doxid-structugo__control__type_1a6a3ba03a92a96dfc5cd1c094e1e3bf87:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` next_interval_selection

how is the next interval for examination chosen:

* 1 = traditional

* 2 = local_improvement

.. index:: pair: variable; refine_with_newton
.. _doxid-structugo__control__type_1a6618fb3c3e71213bde65602fc84b92ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` refine_with_newton

try refine_with_newton Newton steps from the vacinity of the global minimizer to try to improve the estimate

.. index:: pair: variable; alive_unit
.. _doxid-structugo__control__type_1a3fc6359d77a53a63d57ea600b51eac13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structugo__control__type_1ac631699a26f321b14dbed37115f3c006:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; stop_length
.. _doxid-structugo__control__type_1a6bf05a14c29051133abf2e66de24e460:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` stop_length

overall convergence tolerances. The iteration will terminate when the step is less than .stop_length

.. index:: pair: variable; small_g_for_newton
.. _doxid-structugo__control__type_1af8026b35c403d956d877121aa8ec4e7c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` small_g_for_newton

if the absolute value of the gradient is smaller than small_g_for_newton, the next evaluation point may be at a Newton estimate of a local minimizer

.. index:: pair: variable; small_g
.. _doxid-structugo__control__type_1a7c2cf28f2ca5d11f2a013be0a3661e69:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` small_g

if the absolute value of the gradient at the end of the interval search is smaller than small_g, no Newton serach is necessary

.. index:: pair: variable; obj_sufficient
.. _doxid-structugo__control__type_1a307b3c1c0f3796e8195b2339b6082b3b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` obj_sufficient

stop if the objective function is smaller than a specified value

.. index:: pair: variable; global_lipschitz_constant
.. _doxid-structugo__control__type_1a1a37c2054aa6a285605bbfa6f0d5b73b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` global_lipschitz_constant

the global Lipschitz constant for the gradient (-ve means unknown)

.. index:: pair: variable; reliability_parameter
.. _doxid-structugo__control__type_1a73f576b3c9911d11606f473feb090825:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` reliability_parameter

the reliability parameter that is used to boost insufficiently large estimates of the Lipschitz constant (-ve means that default values will be chosen depending on whether second derivatives are provided or not)

.. index:: pair: variable; lipschitz_lower_bound
.. _doxid-structugo__control__type_1aa114fdc06a2a81b1274d165448caa99e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` lipschitz_lower_bound

a lower bound on the Lipscitz constant for the gradient (not zero unless the function is constant)

.. index:: pair: variable; cpu_time_limit
.. _doxid-structugo__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structugo__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; second_derivative_available
.. _doxid-structugo__control__type_1ab91fc45050afb092b9f27ab2910b90d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool second_derivative_available

if .second_derivative_available is true, the user must provide them when requested. The package is generally more effective if second derivatives are available.

.. index:: pair: variable; space_critical
.. _doxid-structugo__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structugo__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structugo__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char prefix[31]

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

