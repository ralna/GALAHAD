.. index:: pair: struct; clls_control_type
.. _doxid-structclls__control__type:

clls_control_type structure
---------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct clls_control_type{T,INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          start_print::INT
          stop_print::INT
          maxit::INT
          infeas_max::INT
          muzero_fixed::INT
          restore_problem::INT
          indicator_type::INT
          arc::INT
          series_order::INT
          sif_file_device::INT
          qplib_file_device::INT
          infinity::T
          stop_abs_p::T
          stop_rel_p::T
          stop_abs_d::T
          stop_rel_d::T
          stop_abs_c::T
          stop_rel_c::T
          prfeas::T
          dufeas::T
          muzero::T
          tau::T
          gamma_c::T
          gamma_f::T
          reduce_infeas::T
          identical_bounds_tol::T
          mu_pounce::T
          indicator_tol_p::T
          indicator_tol_pd::T
          indicator_tol_tapia::T
          cpu_time_limit::T
          clock_time_limit::T
          remove_dependencies::Bool
          treat_zero_bounds_as_general::Bool
          treat_separable_as_general::Bool
          just_feasible::Bool
          getdua::Bool
          puiseux::Bool
          every_order::Bool
          feasol::Bool
          balance_initial_complentarity::Bool
          crossover::Bool
          reduced_pounce_system::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          generate_sif_file::Bool
          generate_qplib_file::Bool
          sif_file_name::NTuple{31,Cchar}
          qplib_file_name::NTuple{31,Cchar}
          prefix::NTuple{31,Cchar}
          fdc_control::fdc_control_type{T,INT}
          sls_control::sls_control_type{T,INT}
          sls_pounce_control::sls_control_type{T,INT}
          fit_control::fit_control_type{INT}
          roots_control::roots_control_type{T,INT}
          cro_control::cro_control_type{T,INT}

.. _details-structclls__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structclls__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structclls__control__type_1a11614f44ef4d939bdd984953346a7572:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structclls__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structclls__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

* $\leq$ 0 gives no output,

* = 1 gives a one-line summary for every iteration,

* = 2 gives a summary of the inner iteration for each iteration,

* $\geq$ 3 gives increasingly verbose (debugging) output

.. index:: pair: variable; start_print
.. _doxid-structclls__control__type_1ae0eb21dc79b53664e45ce07c9109b3aa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT start_print

any printing will start on this iteration

.. index:: pair: variable; stop_print
.. _doxid-structclls__control__type_1a9a3d9960a04602d2a18009c82ae2124e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT stop_print

any printing will stop on this iteration

.. index:: pair: variable; maxit
.. _doxid-structclls__control__type_1ab717630b215f0362699acac11fb3652c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT maxit

at most maxit inner iterations are allowed

.. index:: pair: variable; infeas_max
.. _doxid-structclls__control__type_1af7d33b88b67b2366d7e2df31d42083a9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT infeas_max

the number of iterations for which the overall infeasibility of the problem is not reduced by at least a factor .reduce_infeas before the problem is flagged as infeasible (see reduce_infeas)

.. index:: pair: variable; muzero_fixed
.. _doxid-structclls__control__type_1ab62ab11b934e2380467d5bafe2aaacfb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT muzero_fixed

the initial value of the barrier parameter will not be changed for the first muzero_fixed iterations

.. index:: pair: variable; restore_problem
.. _doxid-structclls__control__type_1a19f10912888ac7ebd153fb21baaeaefa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT restore_problem

indicate whether and how much of the input problem should be restored on output. Possible values are

* 0 nothing restored

* 1 scalar and vector parameters

* 2 all parameters

.. index:: pair: variable; indicator_type
.. _doxid-structclls__control__type_1a5abba51271587463f528b0cbdb478141:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT indicator_type

specifies the type of indicator function used. Possible values are

* 1 primal indicator: a constraint is active if and only if the distance to its nearest bound $\leq$.indicator_p_tol

* 2 primal-dual indicator: a constraint is active if and only if the distance to its nearest bound $\leq$.indicator_tol_pd \* size of corresponding multiplier

* 3 primal-dual indicator: a constraint is active if and only if the distance to its nearest bound $\leq$.indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; arc
.. _doxid-structclls__control__type_1a30b5726116ac85ea64777e5d6e333894:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT arc

which residual trajectory should be used to aim from the current iterate to the solution. Possible values are

* 1 the Zhang linear residual trajectory

* 2 the Zhao-Sun quadratic residual trajectory

* 3 the Zhang arc ultimately switching to the Zhao-Sun residual trajectory

* 4 the mixed linear-quadratic residual trajectory

* 5 the Zhang arc ultimately switching to the mixed linear-quadratic residual trajectory

.. index:: pair: variable; series_order
.. _doxid-structclls__control__type_1a67cdd80087746b1190369ae6ed303b25:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT series_order

the order of (Taylor/Puiseux) series to fit to the path data

.. index:: pair: variable; sif_file_device
.. _doxid-structclls__control__type_1a65c6f8382f1e75cd0b8abd5d148188d0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sif_file_device

specifies the unit number to write generated SIF file describing the current problem

.. index:: pair: variable; qplib_file_device
.. _doxid-structclls__control__type_1a580c343e54a25a2d687782410c9b6917:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT qplib_file_device

specifies the unit number to write generated QPLIB file describing the current problem

.. index:: pair: variable; infinity
.. _doxid-structclls__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

any bound larger than infinity in modulus will be regarded as infinite

.. index:: pair: variable; stop_abs_p
.. _doxid-structclls__control__type_1a3749f9dcb2eeb60815a18c85a7d7d440:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_abs_p

the required absolute and relative accuracies for the primal infeasibility

.. index:: pair: variable; stop_rel_p
.. _doxid-structclls__control__type_1a068065b9d50d5b222fbc8710d530bd9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_p

see stop_abs_p

.. index:: pair: variable; stop_abs_d
.. _doxid-structclls__control__type_1a67b6a7be5dbfa34e9db4c960943fe31f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_abs_d

the required absolute and relative accuracies for the dual infeasibility

.. index:: pair: variable; stop_rel_d
.. _doxid-structclls__control__type_1ab479c27e2044c310e8d0c86869ea2307:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_d

see stop_abs_d

.. index:: pair: variable; stop_abs_c
.. _doxid-structclls__control__type_1a1ed92b5ffb35957c5a8a0e657e312820:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_abs_c

the required absolute and relative accuracies for the complementarity

.. index:: pair: variable; stop_rel_c
.. _doxid-structclls__control__type_1a5c131e3b6061c09670e9c1959b6585a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T stop_rel_c

see stop_abs_c

.. index:: pair: variable; prfeas
.. _doxid-structclls__control__type_1a09a0a5771a0300078ebe5f344ef4e492:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T prfeas

initial primal variables will not be closer than .prfeas from their bounds

.. index:: pair: variable; dufeas
.. _doxid-structclls__control__type_1a2ac34c5838499ed4992037655f52134a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T dufeas

initial dual variables will not be closer than .dufeas from their bounds

.. index:: pair: variable; muzero
.. _doxid-structclls__control__type_1a8f4661dd5869e1555ba329b4bd535b4d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T muzero

the initial value of the barrier parameter. If muzero is not positive, it will be reset to an appropriate value

.. index:: pair: variable; tau
.. _doxid-structclls__control__type_1aa6fb9757f95c75d1a32c5132e939d238:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T tau

the weight attached to primal-dual infeasibility compared to complementa when assessing step acceptance

.. index:: pair: variable; gamma_c
.. _doxid-structclls__control__type_1a80165efc96729e34ab1ae75223dac718:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T gamma_c

individual complementarities will not be allowed to be smaller than gamma_c times the average value

.. index:: pair: variable; gamma_f
.. _doxid-structclls__control__type_1a9ce8b2b646d97d4f0c1485bd8842f198:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T gamma_f

the average complementarity will not be allowed to be smaller than gamma_f times the primal/dual infeasibility

.. index:: pair: variable; reduce_infeas
.. _doxid-structclls__control__type_1aaac67273a340e9f96523583bfdce4c59:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T reduce_infeas

if the overall infeasibility of the problem is not reduced by at least a factor .reduce_infeas over .infeas_max iterations, the problem is flagged as infeasible (see infeas_max)

.. index:: pair: variable; identical_bounds_tol
.. _doxid-structclls__control__type_1abc74ac9bbf6375075f8943aac6ee09e4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T identical_bounds_tol

any pair of constraint bounds $(c_l,c_u)$ or $(x_l,x_u)$ that are closer than .identical_bounds_tol will be reset to the average of their values

.. index:: pair: variable; mu_pounce
.. _doxid-structclls__control__type_1a32eb4d353d409b46521eb28008a74c36:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_pounce

start terminal extrapolation when mu reaches mu_pounce

.. index:: pair: variable; indicator_tol_p
.. _doxid-structclls__control__type_1a9f245bb23cea009fc7a95d86ebe57ddd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T indicator_tol_p

if .indicator_type = 1, a constraint/bound will be deemed to be active if and only if the distance to its nearest bound $\leq$.indicator_p_tol

.. index:: pair: variable; indicator_tol_pd
.. _doxid-structclls__control__type_1aa131ce5e639e5601d1b61fb540ac7187:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T indicator_tol_pd

if .indicator_type = 2, a constraint/bound will be deemed to be active if and only if the distance to its nearest bound $\leq$.indicator_tol_pd \* size of corresponding multiplier

.. index:: pair: variable; indicator_tol_tapia
.. _doxid-structclls__control__type_1abf4fb7dcdbaa9f729af1f063d357000a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T indicator_tol_tapia

if .indicator_type = 3, a constraint/bound will be deemed to be active if and only if the distance to its nearest bound $\leq$.indicator_tol_tapia \* distance to same bound at previous iteration

.. index:: pair: variable; cpu_time_limit
.. _doxid-structclls__control__type_1a52f14ff3f85e6805f2373eef5d0f3dfd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T cpu_time_limit

the maximum CPU time allowed (-ve means infinite)

.. index:: pair: variable; clock_time_limit
.. _doxid-structclls__control__type_1ab05d7c2b06d3a9fb085fa3739501d1c8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time_limit

the maximum elapsed clock time allowed (-ve means infinite)

.. index:: pair: variable; remove_dependencies
.. _doxid-structclls__control__type_1ae17a6b550239434c639239ddf45bc1ad:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool remove_dependencies

the equality constraints will be preprocessed to remove any linear dependencies if true

.. index:: pair: variable; treat_zero_bounds_as_general
.. _doxid-structclls__control__type_1a59b23877e9c8cb49f1f6261e24660295:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool treat_zero_bounds_as_general

any problem bound with the value zero will be treated as if it were a general value if true

.. index:: pair: variable; treat_separable_as_general
.. _doxid-structclls__control__type_1af120d649c006553a2388ef56acc9099a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool treat_separable_as_general

if .just_feasible is true, the algorithm will stop as soon as a feasible point is found. Otherwise, the optimal solution to the problem will be found

.. index:: pair: variable; just_feasible
.. _doxid-structclls__control__type_1a1337f1d22b070690c78f25f7ecaf1e96:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool just_feasible

if .treat_separable_as_general, is true, any separability in the problem structure will be ignored

.. index:: pair: variable; getdua
.. _doxid-structclls__control__type_1ae31443582be206db6b858b35e1fff00b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool getdua

if .getdua, is true, advanced initial values are obtained for the dual variables

.. index:: pair: variable; puiseux
.. _doxid-structclls__control__type_1a444d111a9f28a15760d75b6ff7eb0131:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool puiseux

decide between Puiseux and Taylor series approximations to the arc

.. index:: pair: variable; every_order
.. _doxid-structclls__control__type_1a64cb389e65df8d4add9bc97debd32c69:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool every_order

try every order of series up to series_order?

.. index:: pair: variable; feasol
.. _doxid-structclls__control__type_1a2d28372a45268cac881a4586c4e875d6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasol

if .feasol is true, the final solution obtained will be perturbed so that variables close to their bounds are moved onto these bounds

.. index:: pair: variable; balance_initial_complentarity
.. _doxid-structclls__control__type_1a88bfd9dc0be7872a0bc1ae611d4d1173:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool balance_initial_complentarity

if .balance_initial_complentarity is true, the initial complemetarity is required to be balanced

.. index:: pair: variable; crossover
.. _doxid-structclls__control__type_1a479e35eaf4aeb8b4d0c2d5fe2e4196c4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool crossover

if .crossover is true, cross over the solution to one defined by linearly-independent constraints if possible

.. index:: pair: variable; reduced_pounce_system
.. _doxid-structclls__control__type_1a479e35eaf4aeb8b4d0c2d5fe2e4196c5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool reduced_pounce_system

if .reduced_pounce_system is true,  eliminate fixed variables when solving the linear system required by the attempted pounce to the solution

.. index:: pair: variable; space_critical
.. _doxid-structclls__control__type_1a957fc1f4f26eeef3b0951791ff972e8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structclls__control__type_1a58a2c67fad6e808e8365eff67700cba5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; generate_sif_file
.. _doxid-structclls__control__type_1aa75b3a16d146c0d7ad57bf9817033843:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_sif_file

if .generate_sif_file is .true. if a SIF file describing the current problem is to be generated

.. index:: pair: variable; generate_qplib_file
.. _doxid-structclls__control__type_1ad226b26090e539cf47228ba5ec7dc08d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool generate_qplib_file

if .generate_qplib_file is .true. if a QPLIB file describing the current problem is to be generated

.. index:: pair: variable; sif_file_name
.. _doxid-structclls__control__type_1aaa95e830b709da79d9790471bab54193:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} sif_file_name

name of generated SIF file containing input problem

.. index:: pair: variable; qplib_file_name
.. _doxid-structclls__control__type_1a3d36354e9f30d642f8b081ba85d777d3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} qplib_file_name

name of generated QPLIB file containing input problem

.. index:: pair: variable; prefix
.. _doxid-structclls__control__type_1a1dc05936393ba705f516a0c275df4ffc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

.. index:: pair: variable; fdc_control
.. _doxid-structclls__control__type_1a7bef6e4f678e16a4dcdc40677efddd80:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_control_type<doxid-structfdc__control__type>` fdc_control

control parameters for FDC

.. index:: pair: variable; sls_control
.. _doxid-structclls__control__type_1a04ba974b3c8d21137deb070d0e8dfc3a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_control

control parameters for SLS

.. index:: pair: variable; sls_pounce_control
.. _doxid-structclls__control__type_1a04ba974b3c8d21137deb070d0e8dfc3b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_control_type<doxid-structsls__control__type>` sls_pounce_control

control parameters for SLS pounce

.. index:: pair: variable; fit_control
.. _doxid-structclls__control__type_1a4b2e99077c510333b6b2a6f0b59114b9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fit_control_type<doxid-structfit__control__type>` fit_control

control parameters for FIT

.. index:: pair: variable; roots_control
.. _doxid-structclls__control__type_1a08df6708e7b5364ff3e8fbde29f29014:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`roots_control_type<doxid-structroots__control__type>` roots_control

control parameters for ROOTS

.. index:: pair: variable; cro_control
.. _doxid-structclls__control__type_1a7fbe482405119bceefb8480356d6bd43:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`cro_control_type<doxid-structcro__control__type>` cro_control

control parameters for CRO
