.. index:: pair: table; presolve_inform_type
.. _doxid-structpresolve__inform__type:

presolve_inform_type structure
------------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_presolve.h>
	
	struct presolve_inform_type {
		// fields
	
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status<doxid-structpresolve__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status_continue<doxid-structpresolve__inform__type_1ab9d0ea9275be048bd854d13bd68b06d1>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`status_continued<doxid-structpresolve__inform__type_1ae9021be7a578e777cc20e3cd5f0ab51d>`;
		:ref:`ipc_<doxid-galahad__ipc_8h_>` :ref:`nbr_transforms<doxid-structpresolve__inform__type_1ac678d67797411ebadaab2a5e07f62e8a>`;
		char :ref:`message<doxid-structpresolve__inform__type_1ad3ed7aeb962cfe24345625354d733206>`[3][81];
	};
.. _details-structpresolve__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structpresolve__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` status

The presolve exit condition. It can take the following values (symbol in parentheses is the related Fortran code):

* **0**

  (OK) successful exit;

* **1**

  (MAX_NBR_TRANSF) the maximum number of problem transformation has been
  reached NOTE: this exit is not really an error, since the problem can
  nevertheless be permuted and solved. It merely signals that further
  problem reduction could possibly be obtained with a larger value of
  the parameter ``control.max_nbr_transforms``

* **-1**

  (MEMORY_FULL) memory allocation failed

* **-2**

  (FILE_NOT_OPENED) a file intended for saving problem transformations
  could not be opened;

* **-3**

  (COULD_NOT_WRITE) an IO error occurred while saving transformations on
  the relevant disk file;

* **-4**

  (TOO_FEW_BITS_PER_BYTE) an integer contains less than NBRH + 1 bits.

* **-21**

  (PRIMAL_INFEASIBLE) the problem is primal infeasible;

* **-22**

  (DUAL_INFEASIBLE) the problem is dual infeasible;

* **-23**

  (WRONG_G_DIMENSION) the dimension of the gradient is incompatible with
  the problem dimension;

* **-24**

  (WRONG_HVAL_DIMENSION) the dimension of the vector containing the
  entries of the Hessian is erroneously specified;

* **-25**

  (WRONG_HPTR_DIMENSION) the dimension of the vector containing the
  addresses of the first entry of each Hessian row is erroneously
  specified;

* **-26**

  (WRONG_HCOL_DIMENSION) the dimension of the vector containing the
  column indices of the nonzero Hessian entries is erroneously
  specified;

* **-27**

  (WRONG_HROW_DIMENSION) the dimension of the vector containing the row
  indices of the nonzero Hessian entries is erroneously specified;

* **-28**

  (WRONG_AVAL_DIMENSION) the dimension of the vector containing the
  entries of the Jacobian is erroneously specified;

* **-29**

  (WRONG_APTR_DIMENSION) the dimension of the vector containing the
  addresses of the first entry of each Jacobian row is erroneously
  specified;

* **-30**

  (WRONG_ACOL_DIMENSION) the dimension of the vector containing the
  column indices of the nonzero Jacobian entries is erroneously
  specified;

* **-31**

  (WRONG_AROW_DIMENSION) the dimension of the vector containing the row
  indices of the nonzero Jacobian entries is erroneously specified;

* **-32**

  (WRONG_X_DIMENSION) the dimension of the vector of variables is
  incompatible with the problem dimension;

* **-33**

  (WRONG_XL_DIMENSION) the dimension of the vector of lower bounds on
  the variables is incompatible with the problem dimension;

* **-34**

  (WRONG_XU_DIMENSION) the dimension of the vector of upper bounds on
  the variables is incompatible with the problem dimension;

* **-35**

  (WRONG_Z_DIMENSION) the dimension of the vector of dual variables is
  incompatible with the problem dimension;

* **-36**

  (WRONG_ZL_DIMENSION) the dimension of the vector of lower bounds on
  the dual variables is incompatible with the problem dimension;

* **-37**

  (WRONG_ZU_DIMENSION) the dimension of the vector of upper bounds on
  the dual variables is incompatible with the problem dimension;

* **-38**

  (WRONG_C_DIMENSION) the dimension of the vector of constraints values
  is incompatible with the problem dimension;

* **-39**

  (WRONG_CL_DIMENSION) the dimension of the vector of lower bounds on
  the constraints is incompatible with the problem dimension;

* **-40**

  (WRONG_CU_DIMENSION) the dimension of the vector of upper bounds on
  the constraints is incompatible with the problem dimension;

* **-41**

  (WRONG_Y_DIMENSION) the dimension of the vector of multipliers values
  is incompatible with the problem dimension;

* **-42**

  (WRONG_YL_DIMENSION) the dimension of the vector of lower bounds on
  the multipliers is incompatible with the problem dimension;

* **-43**

  (WRONG_YU_DIMENSION) the dimension of the vector of upper bounds on
  the multipliers is incompatible with the problem dimension;

* **-44**

  (STRUCTURE_NOT_SET) the problem structure has not been set or has been
  cleaned up before an attempt to analyze;

* **-45**

  (PROBLEM_NOT_ANALYZED) the problem has not been analyzed before an
  attempt to permute it;

* **-46**

  (PROBLEM_NOT_PERMUTED) the problem has not been permuted or fully
  reduced before an attempt to restore it

* **-47**

  (H_MISSPECIFIED) the column indices of a row of the sparse Hessian are
  not in increasing order, in that they specify an entry above the
  diagonal;

* **-48**

  (CORRUPTED_SAVE_FILE) one of the files containing saved problem
  transformations has been corrupted between writing and reading;

* **-49**

  (WRONG_XS_DIMENSION) the dimension of the vector of variables' status
  is incompatible with the problem dimension;

* **-50**

  (WRONG_CS_DIMENSION) the dimension of the vector of constraints'
  status is incompatible with the problem dimension;

* **-52**

  (WRONG_N) the problem does not contain any (active) variable;

* **-53**

  (WRONG_M) the problem contains a negative number of constraints;

* **-54**

  (SORT_TOO_LONG) the vectors are too long for the sorting routine;

* **-55**

  (X_OUT_OF_BOUNDS) the value of a variable that is obtained by
  substitution from a constraint is incoherent with the variable's
  bounds. This may be due to a relatively loose accuracy on the linear
  constraints. Try to increase control.c_accuracy.

* **-56**

  (X_NOT_FEASIBLE) the value of a constraint that is obtained by
  recomputing its value on input of ``presolve_restore_solution`` from
  the current x is incompatible with its declared value or its
  bounds. This may caused the restored problem to be infeasible.

* **-57**

  (Z_NOT_FEASIBLE) the value of a dual variable that is obtained by
  recomputing its value on input to ``presolve_restore_solution``
  (assuming dual feasibility) from the current values of $(x, y, z)$ is
  incompatible with its declared value. This may caused the restored
  problem to be infeasible or suboptimal.

* **-58**

  (Z_CANNOT_BE_ZEROED) a dual variable whose value is nonzero because
  the corresponding primal is at an artificial bound cannot be zeroed
  while maintaining dual feasibility (on restoration). This can happen
  when $( x, y, z)$ on input of RESTORE are not (sufficiently) optimal.

* **-60**

  (UNRECOGNIZED_KEYWORD) a keyword was not recognized in the analysis of
  the specification file

* **-61**

  (UNRECOGNIZED_VALUE) a value was not recognized in the analysis of the
  specification file

* **-63**

  (G_NOT_ALLOCATED) the vector G has not been allocated although it has
  general values

* **-64**

  (C_NOT_ALLOCATED) the vector C has not been allocated although m > 0

* **-65**

  (AVAL_NOT_ALLOCATED) the vector A.val has not been allocated although
  m > 0

* **-66**

  (APTR_NOT_ALLOCATED) the vector A.ptr has not been allocated although
  m > 0 and A is stored in row-wise sparse format

* **-67**

  (ACOL_NOT_ALLOCATED) the vector A.col has not been allocated although
  m > 0 and A is stored in row-wise sparse format or sparse coordinate
  format

* **-68**

  (AROW_NOT_ALLOCATED) the vector A.row has not been allocated although
  m > 0 and A is stored in sparse coordinate format

* **-69**

  (HVAL_NOT_ALLOCATED) the vector H.val has not been allocated although
  H.ne > 0

* **-70**

  (HPTR_NOT_ALLOCATED) the vector H.ptr has not been allocated although
  H.ne > 0 and H is stored in row-wise sparse format

* **-71**

  (HCOL_NOT_ALLOCATED) the vector H.col has not been allocated although
  H.ne > 0 and H is stored in row-wise sparse format or sparse
  coordinate format

* **-72**

  (HROW_NOT_ALLOCATED) the vector H.row has not been allocated although
  H.ne > 0 and A is stored in sparse coordinate format

* **-73**

  (WRONG_ANE) incompatible value of A_ne

* **-74**

  (WRONG_HNE) incompatible value of H_ne

.. index:: pair: variable; nbr_transforms
.. _doxid-structpresolve__inform__type_1ac678d67797411ebadaab2a5e07f62e8a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`ipc_<doxid-galahad__ipc_8h_>` nbr_transforms

The final number of problem transformations, as reported to the user at exit.

.. index:: pair: variable; message
.. _doxid-structpresolve__inform__type_1ad3ed7aeb962cfe24345625354d733206:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char message[3][81]

A few lines containing a description of the exit condition on exit of PRESOLVE, typically including more information than indicated in the description of control.status above. It is printed out on device errout at the end of execution if control.print_level >= 1.

