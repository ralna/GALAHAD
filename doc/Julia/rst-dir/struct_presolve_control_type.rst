.. index:: pair: table; presolve_control_type
.. _doxid-structpresolve__control__type:

presolve_control_type structure
-------------------------------

.. toctree::
	:hidden:

control derived type as a Julia structureure :ref:`More...<details-structpresolve__control__type>`


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct presolve_control_type{T,INT}
          f_indexing::Bool
          termination::INT
          max_nbr_transforms::INT
          max_nbr_passes::INT
          c_accuracy::T
          z_accuracy::T
          infinity::T
          out::INT
          errout::INT
          print_level::INT
          dual_transformations::Bool
          redundant_xc::Bool
          primal_constraints_freq::INT
          dual_constraints_freq::INT
          singleton_columns_freq::INT
          doubleton_columns_freq::INT
          unc_variables_freq::INT
          dependent_variables_freq::INT
          sparsify_rows_freq::INT
          max_fill::INT
          transf_file_nbr::INT
          transf_buffer_size::INT
          transf_file_status::INT
          transf_file_name::NTuple{31,Cchar}
          y_sign::INT
          inactive_y::INT
          z_sign::INT
          inactive_z::INT
          final_x_bounds::INT
          final_z_bounds::INT
          final_c_bounds::INT
          final_y_bounds::INT
          check_primal_feasibility::INT
          check_dual_feasibility::INT
          pivot_tol::T
          min_rel_improve::T
          max_growth_factor::T

.. _details-structpresolve__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structpresolve__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; termination
.. _doxid-structpresolve__control__type_termination:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT termination

Determines the strategy for terminating the presolve analysis. Possible values are:

* 1 presolving is continued as long as one of the sizes of the problem (n, m, a_ne, or h_ne) is being reduced;

* 2 presolving is continued as long as problem transformations remain possible. NOTE: the maximum number of analysis passes (control.max_nbr_passes) and the maximum number of problem transformations (control.max_nbr_transforms) set an upper limit on the presolving effort irrespective of the choice of control.termination. The only effect of this latter parameter is to allow for early termination.

.. index:: pair: variable; max_nbr_transforms
.. _doxid-structpresolve__control__type_max_nbr_transforms:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_nbr_transforms

The maximum number of problem transformations, cumulated over all calls to ``presolve``.

.. index:: pair: variable; max_nbr_passes
.. _doxid-structpresolve__control__type_max_nbr_passes:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_nbr_passes

The maximum number of analysis passes for problem analysis during a single call of ``presolve_transform_problem``.

.. index:: pair: variable; c_accuracy
.. _doxid-structpresolve__control__type_c_accuracy:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T c_accuracy

The relative accuracy at which the general linear constraints are satisfied at the exit of the solver. Note that this value is not used before the restoration of the problem.

.. index:: pair: variable; z_accuracy
.. _doxid-structpresolve__control__type_z_accuracy:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T z_accuracy

The relative accuracy at which the dual feasibility constraints are satisfied at the exit of the solver. Note that this value is not used before the restoration of the problem.

.. index:: pair: variable; infinity
.. _doxid-structpresolve__control__type_infinity:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T infinity

The value beyond which a number is deemed equal to plus infinity (minus infinity being defined as its opposite)

.. index:: pair: variable; out
.. _doxid-structpresolve__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

The unit number associated with the device used for printout.

.. index:: pair: variable; errout
.. _doxid-structpresolve__control__type_errout:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT errout

The unit number associated with the device used for error ouput.

.. index:: pair: variable; print_level
.. _doxid-structpresolve__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

The level of printout requested by the user. Can take the values:

* 0 no printout is produced

* 1 only reports the major steps in the analysis

* 2 reports the identity of each problem transformation

* 3 reports more details

* 4 reports lots of information.

* 5 reports a completely silly amount of information

.. index:: pair: variable; dual_transformations
.. _doxid-structpresolve__control__type_dual_transformations:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool dual_transformations

true if dual transformations of the problem are allowed. Note that this implies that the reduced problem is solved accurately (for the dual feasibility condition to hold) as to be able to restore the problem to the original constraints and variables. false prevents dual transformations to be applied, thus allowing for inexact solution of the reduced problem. The setting of this control parameter overides that of get_z, get_z_bounds, get_y, get_y_bounds, dual_constraints_freq, singleton_columns_freq, doubleton_columns_freq, z_accuracy, check_dual_feasibility.

.. index:: pair: variable; redundant_xc
.. _doxid-structpresolve__control__type_redundant_xc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool redundant_xc

true if the redundant variables and constraints (i.e. variables that do not appear in the objective function and appear with a consistent sign in the constraints) are to be removed with their associated constraints before other transformations are attempted.

.. index:: pair: variable; primal_constraints_freq
.. _doxid-structpresolve__control__type_primal_constraints_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT primal_constraints_freq

The frequency of primal constraints analysis in terms of presolving passes. A value of j = 2 indicates that primal constraints are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; dual_constraints_freq
.. _doxid-structpresolve__control__type_dual_constraints_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dual_constraints_freq

The frequency of dual constraints analysis in terms of presolving passes. A value of j = 2 indicates that dual constraints are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; singleton_columns_freq
.. _doxid-structpresolve__control__type_singleton_columns_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT singleton_columns_freq

The frequency of singleton column analysis in terms of presolving passes. A value of j = 2 indicates that singleton columns are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; doubleton_columns_freq
.. _doxid-structpresolve__control__type_doubleton_columns_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT doubleton_columns_freq

The frequency of doubleton column analysis in terms of presolving passes. A value of j indicates that doubleton columns are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; unc_variables_freq
.. _doxid-structpresolve__control__type_unc_variables_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT unc_variables_freq

The frequency of the attempts to fix linearly unconstrained variables, expressed in terms of presolving passes. A value of j = 2 indicates that attempts are made every 2 presolving passes. A zero value indicates that no attempt is ever made.

.. index:: pair: variable; dependent_variables_freq
.. _doxid-structpresolve__control__type_dependent_variables_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT dependent_variables_freq

The frequency of search for dependent variables in terms of presolving passes. A value of j = 2 indicates that dependent variables are searched for every 2 presolving passes. A zero value indicates that they are never searched for.

.. index:: pair: variable; sparsify_rows_freq
.. _doxid-structpresolve__control__type_sparsify_rows_freq:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT sparsify_rows_freq

The frequency of the attempts to make A sparser in terms of presolving passes. A value of j = 2 indicates that attempts are made every 2 presolving passes. A zero value indicates that no attempt is ever made.

.. index:: pair: variable; max_fill
.. _doxid-structpresolve__control__type_max_fill:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_fill

The maximum percentage of fill in each row of A. Note that this is a row-wise measure: globally fill never exceeds the storage initially used for A, no matter how large control.max_fill is chosen. If max_fill is negative, no limit is put on row fill.

.. index:: pair: variable; transf_file_nbr
.. _doxid-structpresolve__control__type_transf_file_nbr:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT transf_file_nbr

The unit number to be associated with the file(s) used for saving problem transformations on a disk file.

.. index:: pair: variable; transf_buffer_size
.. _doxid-structpresolve__control__type_transf_buffer_size:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT transf_buffer_size

The number of transformations that can be kept in memory at once (that is without being saved on a disk file).

.. index:: pair: variable; transf_file_status
.. _doxid-structpresolve__control__type_transf_file_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT transf_file_status

The exit status of the file where problem transformations are saved:

* 0 the file is not deleted after program termination

* 1 the file is not deleted after program termination

.. index:: pair: variable; transf_file_name
.. _doxid-structpresolve__control__type_transf_file_name:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char transf_file_name[31]

The name of the file (to be) used for storing problem transformation on disk. NOTE: this parameter must be identical for all calls to ``presolve`` following ``presolve_read_specfile``. It can then only be changed after calling presolve_terminate.

.. index:: pair: variable; y_sign
.. _doxid-structpresolve__control__type_y_sign:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT y_sign

Determines the convention of sign used for the multipliers associated with the general linear constraints.

* 1 All multipliers corresponding to active inequality constraints are non-negative for lower bound constraints and non-positive for upper bounds constraints.

* -1 All multipliers corresponding to active inequality constraints are non-positive for lower bound constraints and non-negative for upper bounds constraints.

.. index:: pair: variable; inactive_y
.. _doxid-structpresolve__control__type_inactive_y:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT inactive_y

Determines whether or not the multipliers corresponding to constraints that are inactive at the unreduced point corresponding to the reduced point on input to ``presolve_restore_solution`` must be set to zero. Possible values are: associated with the general linear constraints.

* 0 All multipliers corresponding to inactive inequality constraints are forced to zero, possibly at the expense of deteriorating the dual feasibility condition.

* 1 Multipliers corresponding to inactive inequality constraints are left unaltered.

.. index:: pair: variable; z_sign
.. _doxid-structpresolve__control__type_z_sign:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT z_sign

Determines the convention of sign used for the dual variables associated with the bound constraints.

* 1 All dual variables corresponding to active lower bounds are non-negative, and non-positive for active upper bounds.

* -1 All dual variables corresponding to active lower bounds are non-positive, and non-negative for active upper bounds.

.. index:: pair: variable; inactive_z
.. _doxid-structpresolve__control__type_inactive_z:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT inactive_z

Determines whether or not the dual variables corresponding to bounds that are inactive at the unreduced point corresponding to the reduced point on input to ``presolve_restore_solution`` must be set to zero. Possible values are: associated with the general linear constraints.

* 0: All dual variables corresponding to inactive bounds are forced to zero, possibly at the expense of deteriorating the dual feasibility condition.

* 1 Dual variables corresponding to inactive bounds are left unaltered.

.. index:: pair: variable; final_x_bounds
.. _doxid-structpresolve__control__type_final_x_bounds:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT final_x_bounds

The type of final bounds on the variables returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the variables (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* n must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTE: this parameter must be identical for all calls to presolve (except presolve_initialize).

.. index:: pair: variable; final_z_bounds
.. _doxid-structpresolve__control__type_final_z_bounds:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT final_z_bounds

The type of final bounds on the dual variables returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the dual variables (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* n must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTE: this parameter must be identical for all calls to presolve (except presolve_initialize).

.. index:: pair: variable; final_c_bounds
.. _doxid-structpresolve__control__type_final_c_bounds:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT final_c_bounds

The type of final bounds on the constraints returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the constraints (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* m must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTES: 1) This parameter must be identical for all calls to presolve (except presolve_initialize). 2) If different from 0, its value must be identical to that of control.final_x_bounds.

.. index:: pair: variable; final_y_bounds
.. _doxid-structpresolve__control__type_final_y_bounds:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT final_y_bounds

The type of final bounds on the multipliers returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the multipliers (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* m must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTE: this parameter must be identical for all calls to presolve (except presolve_initialize).

.. index:: pair: variable; check_primal_feasibility
.. _doxid-structpresolve__control__type_check_primal_feasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT check_primal_feasibility

The level of feasibility check (on the values of x) at the start of the restoration phase. This parameter can take the values:

* 0 no check at all;

* 1 the primal constraints are recomputed at x and a message issued if the computed value does not match the input value, or if it is out of bounds (if control.print_level >= 2);

* 2 the same as for 1, but presolve is terminated if an incompatibilty is detected.

.. index:: pair: variable; check_dual_feasibility
.. _doxid-structpresolve__control__type_check_dual_feasibility:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT check_dual_feasibility

The level of dual feasibility check (on the values of x, y and z) at the start of the restoration phase. This parameter can take the values:

* 0 no check at all;

* 1 the dual feasibility condition is recomputed at ( x, y, z ) and a message issued if the computed value does not match the input value (if control.print_level >= 2);

* 2 the same as for 1, but presolve is terminated if an incompatibilty is detected. The last two values imply the allocation of an additional real workspace vector of size equal to the number of variables in the reduced problem.

.. index:: pair: variable; pivot_tol
.. _doxid-structpresolve__control__type_pivot_tol:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pivot_tol

The relative pivot tolerance above which pivoting is considered as numerically stable in transforming the coefficient matrix A. A zero value corresponds to a totally unsafeguarded pivoting strategy (potentially unstable).

.. index:: pair: variable; min_rel_improve
.. _doxid-structpresolve__control__type_min_rel_improve:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T min_rel_improve

The minimum relative improvement in the bounds on x, y and z for a tighter bound on these quantities to be accepted in the course of the analysis. More formally, if lower is the current value of the lower bound on one of the x, y or z, and if new_lower is a tentative tighter lower bound on the same quantity, it is only accepted if.

new_lower >= lower + tol \* MAX( 1, ABS( lower ) ),

where

tol = control.min_rel_improve.

Similarly, a tentative tighter upper bound new_upper only replaces the current upper bound upper if

new_upper <= upper - tol \* MAX( 1, ABS( upper ) ).

Note that this parameter must exceed the machine precision significantly.

.. index:: pair: variable; max_growth_factor
.. _doxid-structpresolve__control__type_max_growth_factor:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T max_growth_factor

The maximum growth factor (in absolute value) that is accepted between the maximum data item in the original problem and any data item in the reduced problem. If a transformation results in this bound being exceeded, the transformation is skipped.

