.. index:: pair: struct; ugo_control_type
.. _doxid-structugo__control__type:

ugo_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ugo_control_type{T,INT}
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          print_gap::INT
          maxit::INT
          initial_points::INT
          storage_increment::INT
          buffer::INT
          lipschitz_estimate_used::INT
          next_interval_selection::INT
          refine_with_newton::INT
          alive_unit::INT
          alive_file::NTuple{31,Cchar}
          stop_length::T
          small_g_for_newton::T
          small_g::T
          obj_sufficient::T
          global_lipschitz_constant::T
          reliability_parameter::T
          lipschitz_lower_bound::T
          cpu_time_limit::T
          clock_time_limit::T
          second_derivative_available::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structugo__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; error
.. _doxid-structugo__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structugo__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structugo__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* 1 a one-line summary for every improvement

* 2 a summary of each iteration

* $\geq$ 3 increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structugo__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structugo__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structugo__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structugo__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

the maximum number of iterations allowed

.. index:: pair: variable; initial_points
.. _doxid-structugo__control__type_initial_points:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT initial_points

the number of initial (uniformly-spaced) evaluation points (<2 reset to 2)

.. index:: pair: variable; storage_increment
.. _doxid-structugo__control__type_storage_increment:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT storage_increment

incremenets of storage allocated (less that 1000 will be reset to 1000)

.. index:: pair: variable; buffer
.. _doxid-structugo__control__type_buffer:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT buffer

unit for any out-of-core writing when expanding arrays

.. index:: pair: variable; lipschitz_estimate_used
.. _doxid-structugo__control__type_lipschitz_estimate_used:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lipschitz_estimate_used

what sort of Lipschitz constant estimate will be used:

* 1 = global contant provided

* 2 = global contant estimated

* 3 = local costants estimated

.. index:: pair: variable; next_interval_selection
.. _doxid-structugo__control__type_next_interval_selection:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT next_interval_selection

how is the next interval for examination chosen:

* 1 = traditional

* 2 = local_improvement

.. index:: pair: variable; refine_with_newton
.. _doxid-structugo__control__type_refine_with_newton:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT refine_with_newton

try refine_with_newton Newton steps from the vacinity of the global minimizer to try to improve the estimate

.. index:: pair: variable; alive_unit
.. _doxid-structugo__control__type_alive_unit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structugo__control__type_alive_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} alive_file

see alive_unit

.. index:: pair: variable; stop_length
.. _doxid-structugo__control__type_stop_length:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_length

overall convergence tolerances. The iteration will terminate when the step is less than .stop_length

.. index:: pair: variable; small_g_for_newton
.. _doxid-structugo__control__type_small_g_for_newton:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T small_g_for_newton

if the absolute value of the gradient is smaller than small_g_for_newton, the next evaluation point may be at a Newton estimate of a local minimizer

.. index:: pair: variable; small_g
.. _doxid-structugo__control__type_small_g:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T small_g

if the absolute value of the gradient at the end of the interval search is smaller than small_g, no Newton serach is necessary

.. index:: pair: variable; obj_sufficient
.. _doxid-structugo__control__type_obj_sufficient:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_sufficient

stop if the objective function is smaller than a specified value

.. index:: pair: variable; global_lipschitz_constant
.. _doxid-structugo__control__type_global_lipschitz_constant:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T global_lipschitz_constant

the global Lipschitz constant for the gradient (-ve means unknown)

.. index:: pair: variable; reliability_parameter
.. _doxid-structugo__control__type_reliability_parameter:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reliability_parameter

the reliability parameter that is used to boost insufficiently large estimates of the Lipschitz constant (-ve means that default values will be chosen depending on whether second derivatives are provided or not)

.. index:: pair: variable; lipschitz_lower_bound
.. _doxid-structugo__control__type_lipschitz_lower_bound:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lipschitz_lower_bound

a lower bound on the Lipscitz constant for the gradient (not zero unless the function is constant)

.. index:: pair: variable; cpu_time_limit
.. _doxid-structugo__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structugo__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; second_derivative_available
.. _doxid-structugo__control__type_second_derivative_available:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool second_derivative_available

if .second_derivative_available is true, the user must provide them when requested. The package is generally more effective if second derivatives are available.

.. index:: pair: variable; space_critical
.. _doxid-structugo__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical is true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structugo__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structugo__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'
