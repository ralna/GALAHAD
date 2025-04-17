.. index:: pair: struct; dgo_control_type
.. _doxid-structdgo__control__type:

dgo_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct dgo_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          print_gap::INT
          maxit::INT
          max_evals::INT
          dictionary_size::INT
          alive_unit::INT
          alive_file::NTuple{31,Cchar}
          infinity::T
          lipschitz_lower_bound::T
          lipschitz_reliability::T
          lipschitz_control::T
          stop_length::T
          stop_f::T
          obj_unbounded::T
          cpu_time_limit::T
          clock_time_limit::T
          hessian_available::Bool
          prune::Bool
          perform_local_optimization::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}
          hash_control::hash_control_type{INT}
          ugo_control::ugo_control_type{T,INT}
          trb_control::trb_control_type{T,INT}

.. _details-structdgo__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structdgo__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structdgo__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structdgo__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structdgo__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* 1 a one-line summary for every improvement

* 2 a summary of each iteration

* $\geq$ 3 increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structdgo__control__type_start_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structdgo__control__type_stop_print:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; print_gap
.. _doxid-structdgo__control__type_print_gap:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_gap

the number of iterations between printing

.. index:: pair: variable; maxit
.. _doxid-structdgo__control__type_maxit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

the maximum number of iterations performed

.. index:: pair: variable; max_evals
.. _doxid-structdgo__control__type_max_evals:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_evals

the maximum number of function evaluations made

.. index:: pair: variable; dictionary_size
.. _doxid-structdgo__control__type_dictionary_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dictionary_size

the size of the initial hash dictionary

.. index:: pair: variable; alive_unit
.. _doxid-structdgo__control__type_alive_unit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structdgo__control__type_alive_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; infinity
.. _doxid-structdgo__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; lipschitz_lower_bound
.. _doxid-structdgo__control__type_lipschitz_lower_bound:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lipschitz_lower_bound

a small positive constant (<= 1e-6) that ensure that the estimted gradient Lipschitz constant is not too small

.. index:: pair: variable; lipschitz_reliability
.. _doxid-structdgo__control__type_lipschitz_reliability:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lipschitz_reliability

the Lipschitz reliability parameter, the Lipschiz constant used will be a factor lipschitz_reliability times the largest value observed

.. index:: pair: variable; lipschitz_control
.. _doxid-structdgo__control__type_lipschitz_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T lipschitz_control

the reliablity control parameter, the actual reliability parameter used will be .lipschitz_reliability

* MAX( 1, n - 1 ) \* .lipschitz_control / iteration

.. index:: pair: variable; stop_length
.. _doxid-structdgo__control__type_stop_length:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_length

the iteration will stop if the length, delta, of the diagonal in the box with the smallest-found objective function is smaller than .stop_length times that of the original bound box, delta_0

.. index:: pair: variable; stop_f
.. _doxid-structdgo__control__type_stop_f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_f

the iteration will stop if the gap between the best objective value found and the smallest lower bound is smaller than .stop_f

.. index:: pair: variable; obj_unbounded
.. _doxid-structdgo__control__type_obj_unbounded:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structdgo__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structdgo__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; hessian_available
.. _doxid-structdgo__control__type_hessian_available:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; prune
.. _doxid-structdgo__control__type_prune:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool prune

should boxes that cannot contain the global minimizer be pruned (i.e., removed from further consideration)?

.. index:: pair: variable; perform_local_optimization
.. _doxid-structdgo__control__type_perform_local_optimization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool perform_local_optimization

should approximate minimizers be impoved by judicious local minimization?

.. index:: pair: variable; space_critical
.. _doxid-structdgo__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structdgo__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structdgo__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; hash_control
.. _doxid-structdgo__control__type_hash_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`hash_control_type<doxid-structhash__control__type>` hash_control

control parameters for HASH

.. index:: pair: variable; ugo_control
.. _doxid-structdgo__control__type_ugo_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ugo_control_type<doxid-structugo__control__type>` ugo_control

control parameters for UGO

.. index:: pair: variable; trb_control
.. _doxid-structdgo__control__type_trb_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trb_control_type<doxid-structtrb__control__type>` trb_control

control parameters for TRB

