.. index:: pair: table; presolve_control_type
.. _doxid-structpresolve__control__type:

presolve_control_type structure
-------------------------------

.. toctree::
	:hidden:

control derived type as a C struct :ref:`More...<details-structpresolve__control__type>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_presolve.h>
	
	struct presolve_control_type {
		// fields
	
		bool :ref:`f_indexing<doxid-structpresolve__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`termination<doxid-structpresolve__control__type_1a8812e14a78de75cb35920f1ca14f8fcb>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_nbr_transforms<doxid-structpresolve__control__type_1ad2cba4a8892265253e3821f2a8398783>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_nbr_passes<doxid-structpresolve__control__type_1ab9b9f9490ee04ad60a88c98bedeb69bf>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`c_accuracy<doxid-structpresolve__control__type_1afb60a9e6d661aebf74d5da10af97233f>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`z_accuracy<doxid-structpresolve__control__type_1af3699b1a6b62c80d06c848f9bf316708>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`infinity<doxid-structpresolve__control__type_1a11a46bd456ea63bac8bdffb056fe98c9>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`out<doxid-structpresolve__control__type_1aa8000eda101cade7c6c4b913fce0cc9c>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`errout<doxid-structpresolve__control__type_1a96f36bbf8aecb8c1df4e9479e0495341>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`print_level<doxid-structpresolve__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81>`;
		bool :ref:`dual_transformations<doxid-structpresolve__control__type_1a622f854458f44802d65ea5c644488b05>`;
		bool :ref:`redundant_xc<doxid-structpresolve__control__type_1a7fff1fe9af8dc83ace001c1f01daca4e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`primal_constraints_freq<doxid-structpresolve__control__type_1ac5198d5a57920d3cbbc44fc43e1f461d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dual_constraints_freq<doxid-structpresolve__control__type_1a5de3600b41511490861e5f5cc52c6c8d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`singleton_columns_freq<doxid-structpresolve__control__type_1a79b5f7f8d67056004a9ec1347eaf0b2a>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`doubleton_columns_freq<doxid-structpresolve__control__type_1a3f84400443972b69eb439c8ddbccd6e4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`unc_variables_freq<doxid-structpresolve__control__type_1a83a0db3aa9212dc0630226a80f23355e>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`dependent_variables_freq<doxid-structpresolve__control__type_1a916449dc4c15cc8c573f65d13a7b1837>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`sparsify_rows_freq<doxid-structpresolve__control__type_1aedb83124a2aeb24018ca314263ea194d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`max_fill<doxid-structpresolve__control__type_1a62b85e62f2dd65b004b9561006447321>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`transf_file_nbr<doxid-structpresolve__control__type_1af309919911cc80fb67ce4309caff53ee>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`transf_buffer_size<doxid-structpresolve__control__type_1a8e55be8b47271c8bfe04e4eb7abe41d9>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`transf_file_status<doxid-structpresolve__control__type_1a3327bff5444eebb4a46aa1671123681f>`;
		char :ref:`transf_file_name<doxid-structpresolve__control__type_1af7814ff832c4c4e7fab56d53c26b4bed>`[31];
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`y_sign<doxid-structpresolve__control__type_1a15a549b266499b20a45e93ce9a91f083>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`inactive_y<doxid-structpresolve__control__type_1a0ddf7a757e8d25df82bbd45c4cc522f4>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`z_sign<doxid-structpresolve__control__type_1aa643ec4b5dfb05f11c4c932158d92a37>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`inactive_z<doxid-structpresolve__control__type_1a02a187d6425a995f970b86ab1ae6deaa>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`final_x_bounds<doxid-structpresolve__control__type_1a980e04d17981a03c6ce9142915baeec6>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`final_z_bounds<doxid-structpresolve__control__type_1a8b0b8e949abf9bb7eae7f6e258a9fadf>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`final_c_bounds<doxid-structpresolve__control__type_1a7797130742a276bfa34e28713c5f69fe>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`final_y_bounds<doxid-structpresolve__control__type_1a8d59a01fe70d5186b185454844aa5388>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`check_primal_feasibility<doxid-structpresolve__control__type_1a63d94a12589b2a15bedc5d4b172563d7>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`check_dual_feasibility<doxid-structpresolve__control__type_1a953e9d14756db97aaecceca97b78f334>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`pivot_tol<doxid-structpresolve__control__type_1a133347eb5f45a24a77b63b4afd4212e8>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`min_rel_improve<doxid-structpresolve__control__type_1a6ff1d4c2c7c9a996e081de4beccebf86>`;
		:ref:`rpc_<doxid-galahad__rpc_8h_>` :ref:`max_growth_factor<doxid-structpresolve__control__type_1ac768d36daebcdaaec3ad82313c45fa64>`;
	};
.. _details-structpresolve__control__type:

detailed documentation
----------------------

control derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structpresolve__control__type_1a6e8421b34d6b85dcb33c1dd0179efbb3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; termination
.. _doxid-structpresolve__control__type_1a8812e14a78de75cb35920f1ca14f8fcb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` termination

Determines the strategy for terminating the presolve analysis. Possible values are:

* 1 presolving is continued as long as one of the sizes of the problem (n, m, a_ne, or h_ne) is being reduced;

* 2 presolving is continued as long as problem transformations remain possible. NOTE: the maximum number of analysis passes (control.max_nbr_passes) and the maximum number of problem transformations (control.max_nbr_transforms) set an upper limit on the presolving effort irrespective of the choice of control.termination. The only effect of this latter parameter is to allow for early termination.

.. index:: pair: variable; max_nbr_transforms
.. _doxid-structpresolve__control__type_1ad2cba4a8892265253e3821f2a8398783:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_nbr_transforms

The maximum number of problem transformations, cumulated over all calls to ``presolve``.

.. index:: pair: variable; max_nbr_passes
.. _doxid-structpresolve__control__type_1ab9b9f9490ee04ad60a88c98bedeb69bf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_nbr_passes

The maximum number of analysis passes for problem analysis during a single call of ``presolve_transform_problem``.

.. index:: pair: variable; c_accuracy
.. _doxid-structpresolve__control__type_1afb60a9e6d661aebf74d5da10af97233f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` c_accuracy

The relative accuracy at which the general linear constraints are satisfied at the exit of the solver. Note that this value is not used before the restoration of the problem.

.. index:: pair: variable; z_accuracy
.. _doxid-structpresolve__control__type_1af3699b1a6b62c80d06c848f9bf316708:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` z_accuracy

The relative accuracy at which the dual feasibility constraints are satisfied at the exit of the solver. Note that this value is not used before the restoration of the problem.

.. index:: pair: variable; infinity
.. _doxid-structpresolve__control__type_1a11a46bd456ea63bac8bdffb056fe98c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` infinity

The value beyond which a number is deemed equal to plus infinity (minus infinity being defined as its opposite)

.. index:: pair: variable; out
.. _doxid-structpresolve__control__type_1aa8000eda101cade7c6c4b913fce0cc9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` out

The unit number associated with the device used for printout.

.. index:: pair: variable; errout
.. _doxid-structpresolve__control__type_1a96f36bbf8aecb8c1df4e9479e0495341:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` errout

The unit number associated with the device used for error ouput.

.. index:: pair: variable; print_level
.. _doxid-structpresolve__control__type_1a12dae630bd8f5d2d00f6a86d652f5c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` print_level

The level of printout requested by the user. Can take the values:

* 0 no printout is produced

* 1 only reports the major steps in the analysis

* 2 reports the identity of each problem transformation

* 3 reports more details

* 4 reports lots of information.

* 5 reports a completely silly amount of information

.. index:: pair: variable; dual_transformations
.. _doxid-structpresolve__control__type_1a622f854458f44802d65ea5c644488b05:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool dual_transformations

true if dual transformations of the problem are allowed. Note that this implies that the reduced problem is solved accurately (for the dual feasibility condition to hold) as to be able to restore the problem to the original constraints and variables. false prevents dual transformations to be applied, thus allowing for inexact solution of the reduced problem. The setting of this control parameter overides that of get_z, get_z_bounds, get_y, get_y_bounds, dual_constraints_freq, singleton_columns_freq, doubleton_columns_freq, z_accuracy, check_dual_feasibility.

.. index:: pair: variable; redundant_xc
.. _doxid-structpresolve__control__type_1a7fff1fe9af8dc83ace001c1f01daca4e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool redundant_xc

true if the redundant variables and constraints (i.e. variables that do not appear in the objective function and appear with a consistent sign in the constraints) are to be removed with their associated constraints before other transformations are attempted.

.. index:: pair: variable; primal_constraints_freq
.. _doxid-structpresolve__control__type_1ac5198d5a57920d3cbbc44fc43e1f461d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` primal_constraints_freq

The frequency of primal constraints analysis in terms of presolving passes. A value of j = 2 indicates that primal constraints are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; dual_constraints_freq
.. _doxid-structpresolve__control__type_1a5de3600b41511490861e5f5cc52c6c8d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dual_constraints_freq

The frequency of dual constraints analysis in terms of presolving passes. A value of j = 2 indicates that dual constraints are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; singleton_columns_freq
.. _doxid-structpresolve__control__type_1a79b5f7f8d67056004a9ec1347eaf0b2a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` singleton_columns_freq

The frequency of singleton column analysis in terms of presolving passes. A value of j = 2 indicates that singleton columns are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; doubleton_columns_freq
.. _doxid-structpresolve__control__type_1a3f84400443972b69eb439c8ddbccd6e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` doubleton_columns_freq

The frequency of doubleton column analysis in terms of presolving passes. A value of j indicates that doubleton columns are analyzed every 2 presolving passes. A zero value indicates that they are never analyzed.

.. index:: pair: variable; unc_variables_freq
.. _doxid-structpresolve__control__type_1a83a0db3aa9212dc0630226a80f23355e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` unc_variables_freq

The frequency of the attempts to fix linearly unconstrained variables, expressed in terms of presolving passes. A value of j = 2 indicates that attempts are made every 2 presolving passes. A zero value indicates that no attempt is ever made.

.. index:: pair: variable; dependent_variables_freq
.. _doxid-structpresolve__control__type_1a916449dc4c15cc8c573f65d13a7b1837:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` dependent_variables_freq

The frequency of search for dependent variables in terms of presolving passes. A value of j = 2 indicates that dependent variables are searched for every 2 presolving passes. A zero value indicates that they are never searched for.

.. index:: pair: variable; sparsify_rows_freq
.. _doxid-structpresolve__control__type_1aedb83124a2aeb24018ca314263ea194d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` sparsify_rows_freq

The frequency of the attempts to make A sparser in terms of presolving passes. A value of j = 2 indicates that attempts are made every 2 presolving passes. A zero value indicates that no attempt is ever made.

.. index:: pair: variable; max_fill
.. _doxid-structpresolve__control__type_1a62b85e62f2dd65b004b9561006447321:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` max_fill

The maximum percentage of fill in each row of A. Note that this is a row-wise measure: globally fill never exceeds the storage initially used for A, no matter how large control.max_fill is chosen. If max_fill is negative, no limit is put on row fill.

.. index:: pair: variable; transf_file_nbr
.. _doxid-structpresolve__control__type_1af309919911cc80fb67ce4309caff53ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` transf_file_nbr

The unit number to be associated with the file(s) used for saving problem transformations on a disk file.

.. index:: pair: variable; transf_buffer_size
.. _doxid-structpresolve__control__type_1a8e55be8b47271c8bfe04e4eb7abe41d9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` transf_buffer_size

The number of transformations that can be kept in memory at once (that is without being saved on a disk file).

.. index:: pair: variable; transf_file_status
.. _doxid-structpresolve__control__type_1a3327bff5444eebb4a46aa1671123681f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` transf_file_status

The exit status of the file where problem transformations are saved:

* 0 the file is not deleted after program termination

* 1 the file is not deleted after program termination

.. index:: pair: variable; transf_file_name
.. _doxid-structpresolve__control__type_1af7814ff832c4c4e7fab56d53c26b4bed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char transf_file_name[31]

The name of the file (to be) used for storing problem transformation on disk. NOTE: this parameter must be identical for all calls to ``presolve`` following ``presolve_read_specfile``. It can then only be changed after calling presolve_terminate.

.. index:: pair: variable; y_sign
.. _doxid-structpresolve__control__type_1a15a549b266499b20a45e93ce9a91f083:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` y_sign

Determines the convention of sign used for the multipliers associated with the general linear constraints.

* 1 All multipliers corresponding to active inequality constraints are non-negative for lower bound constraints and non-positive for upper bounds constraints.

* -1 All multipliers corresponding to active inequality constraints are non-positive for lower bound constraints and non-negative for upper bounds constraints.

.. index:: pair: variable; inactive_y
.. _doxid-structpresolve__control__type_1a0ddf7a757e8d25df82bbd45c4cc522f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` inactive_y

Determines whether or not the multipliers corresponding to constraints that are inactive at the unreduced point corresponding to the reduced point on input to ``presolve_restore_solution`` must be set to zero. Possible values are: associated with the general linear constraints.

* 0 All multipliers corresponding to inactive inequality constraints are forced to zero, possibly at the expense of deteriorating the dual feasibility condition.

* 1 Multipliers corresponding to inactive inequality constraints are left unaltered.

.. index:: pair: variable; z_sign
.. _doxid-structpresolve__control__type_1aa643ec4b5dfb05f11c4c932158d92a37:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` z_sign

Determines the convention of sign used for the dual variables associated with the bound constraints.

* 1 All dual variables corresponding to active lower bounds are non-negative, and non-positive for active upper bounds.

* -1 All dual variables corresponding to active lower bounds are non-positive, and non-negative for active upper bounds.

.. index:: pair: variable; inactive_z
.. _doxid-structpresolve__control__type_1a02a187d6425a995f970b86ab1ae6deaa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` inactive_z

Determines whether or not the dual variables corresponding to bounds that are inactive at the unreduced point corresponding to the reduced point on input to ``presolve_restore_solution`` must be set to zero. Possible values are: associated with the general linear constraints.

* 0: All dual variables corresponding to inactive bounds are forced to zero, possibly at the expense of deteriorating the dual feasibility condition.

* 1 Dual variables corresponding to inactive bounds are left unaltered.

.. index:: pair: variable; final_x_bounds
.. _doxid-structpresolve__control__type_1a980e04d17981a03c6ce9142915baeec6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` final_x_bounds

The type of final bounds on the variables returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the variables (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* n must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTE: this parameter must be identical for all calls to presolve (except presolve_initialize).

.. index:: pair: variable; final_z_bounds
.. _doxid-structpresolve__control__type_1a8b0b8e949abf9bb7eae7f6e258a9fadf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` final_z_bounds

The type of final bounds on the dual variables returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the dual variables (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* n must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTE: this parameter must be identical for all calls to presolve (except presolve_initialize).

.. index:: pair: variable; final_c_bounds
.. _doxid-structpresolve__control__type_1a7797130742a276bfa34e28713c5f69fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` final_c_bounds

The type of final bounds on the constraints returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the constraints (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* m must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTES: 1) This parameter must be identical for all calls to presolve (except presolve_initialize). 2) If different from 0, its value must be identical to that of control.final_x_bounds.

.. index:: pair: variable; final_y_bounds
.. _doxid-structpresolve__control__type_1a8d59a01fe70d5186b185454844aa5388:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` final_y_bounds

The type of final bounds on the multipliers returned by the package. This parameter can take the values:

* 0 the final bounds are the tightest bounds known on the multipliers (at the risk of being redundant with other constraints, which may cause degeneracy);

* 1 the best known bounds that are known to be non-degenerate. This option implies that an additional real workspace of size 2 \* m must be allocated.

* 2 the loosest bounds that are known to keep the problem equivalent to the original problem. This option also implies that an additional real workspace of size 2 \* n must be allocated.

NOTE: this parameter must be identical for all calls to presolve (except presolve_initialize).

.. index:: pair: variable; check_primal_feasibility
.. _doxid-structpresolve__control__type_1a63d94a12589b2a15bedc5d4b172563d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` check_primal_feasibility

The level of feasibility check (on the values of x) at the start of the restoration phase. This parameter can take the values:

* 0 no check at all;

* 1 the primal constraints are recomputed at x and a message issued if the computed value does not match the input value, or if it is out of bounds (if control.print_level >= 2);

* 2 the same as for 1, but presolve is terminated if an incompatibilty is detected.

.. index:: pair: variable; check_dual_feasibility
.. _doxid-structpresolve__control__type_1a953e9d14756db97aaecceca97b78f334:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` check_dual_feasibility

The level of dual feasibility check (on the values of x, y and z) at the start of the restoration phase. This parameter can take the values:

* 0 no check at all;

* 1 the dual feasibility condition is recomputed at ( x, y, z ) and a message issued if the computed value does not match the input value (if control.print_level >= 2);

* 2 the same as for 1, but presolve is terminated if an incompatibilty is detected. The last two values imply the allocation of an additional real workspace vector of size equal to the number of variables in the reduced problem.

.. index:: pair: variable; pivot_tol
.. _doxid-structpresolve__control__type_1a133347eb5f45a24a77b63b4afd4212e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` pivot_tol

The relative pivot tolerance above which pivoting is considered as numerically stable in transforming the coefficient matrix A. A zero value corresponds to a totally unsafeguarded pivoting strategy (potentially unstable).

.. index:: pair: variable; min_rel_improve
.. _doxid-structpresolve__control__type_1a6ff1d4c2c7c9a996e081de4beccebf86:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` min_rel_improve

The minimum relative improvement in the bounds on x, y and z for a tighter bound on these quantities to be accepted in the course of the analysis. More formally, if lower is the current value of the lower bound on one of the x, y or z, and if new_lower is a tentative tighter lower bound on the same quantity, it is only accepted if.

new_lower >= lower + tol \* MAX( 1, ABS( lower ) ),

where

tol = control.min_rel_improve.

Similarly, a tentative tighter upper bound new_upper only replaces the current upper bound upper if

new_upper <= upper - tol \* MAX( 1, ABS( upper ) ).

Note that this parameter must exceed the machine precision significantly.

.. index:: pair: variable; max_growth_factor
.. _doxid-structpresolve__control__type_1ac768d36daebcdaaec3ad82313c45fa64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`rpc_<doxid-galahad__rpc_8h_>` max_growth_factor

The maximum growth factor (in absolute value) that is accepted between the maximum data item in the original problem and any data item in the reduced problem. If a transformation results in this bound being exceeded, the transformation is skipped.

