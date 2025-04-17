.. index:: pair: struct; bgo_control_type
.. _doxid-structbgo__control__type:

bgo_control_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bgo_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          attempts_max::INT
          max_evals::INT
          sampling_strategy::INT
          hypercube_discretization::INT
          alive_unit::INT
          alive_file::NTuple{31,Cchar}
          infinity::T
          obj_unbounded::T
          cpu_time_limit::T
          clock_time_limit::T
          random_multistart::Bool
          hessian_available::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}
          ugo_control::ugo_control_type{T,INT}
          lhs_control::lhs_control_type{INT}
          trb_control::trb_control_type{T,INT}

.. _details-structbgo__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbgo__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbgo__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structbgo__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structbgo__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required. Possible values are:

* $\leq$ 0 no output,

* 1 a one-line summary for every improvement

* 2 a summary of each iteration

* $\geq$ 3 increasingly verbose (debugging) output

.. index:: pair: variable; attempts_max
.. _doxid-structbgo__control__type_attempts_max:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT attempts_max

the maximum number of random searches from the best point found so far

.. index:: pair: variable; max_evals
.. _doxid-structbgo__control__type_max_evals:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_evals

the maximum number of function evaluations made

.. index:: pair: variable; sampling_strategy
.. _doxid-structbgo__control__type_sampling_strategy:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sampling_strategy

sampling strategy used. Possible values are

* 1 uniformly spread

* 2 Latin hypercube sampling

* 3 niformly spread within a Latin hypercube

.. index:: pair: variable; hypercube_discretization
.. _doxid-structbgo__control__type_hypercube_discretization:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT hypercube_discretization

hyper-cube discretization (for sampling stategies 2 and 3)

.. index:: pair: variable; alive_unit
.. _doxid-structbgo__control__type_alive_unit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alive_unit

removal of the file alive_file from unit alive_unit terminates execution

.. index:: pair: variable; alive_file
.. _doxid-structbgo__control__type_alive_file:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char alive_file[31]

see alive_unit

.. index:: pair: variable; infinity
.. _doxid-structbgo__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; obj_unbounded
.. _doxid-structbgo__control__type_obj_unbounded:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj_unbounded

the smallest value the objective function may take before the problem is marked as unbounded

.. index:: pair: variable; cpu_time_limit
.. _doxid-structbgo__control__type_cpu_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structbgo__control__type_clock_time_limit:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; random_multistart
.. _doxid-structbgo__control__type_random_multistart:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool random_multistart

perform random-multistart as opposed to local minimize and probe

.. index:: pair: variable; hessian_available
.. _doxid-structbgo__control__type_hessian_available:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool hessian_available

is the Hessian matrix of second derivatives available or is access only via matrix-vector products?

.. index:: pair: variable; space_critical
.. _doxid-structbgo__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbgo__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structbgo__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; ugo_control
.. _doxid-structbgo__control__type_ugo_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ugo_control_type<doxid-structugo__control__type>` ugo_control

control parameters for UGO

.. index:: pair: variable; lhs_control
.. _doxid-structbgo__control__type_lhs_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lhs_control_type<doxid-structlhs__control__type>` lhs_control

control parameters for LHS

.. index:: pair: variable; trb_control
.. _doxid-structbgo__control__type_trb_control:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`trb_control_type<doxid-structtrb__control__type>` trb_control

control parameters for TRB

