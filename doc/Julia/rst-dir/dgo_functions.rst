.. _global:

callable functions
------------------

.. index:: pair: function; dgo_initialize
.. _doxid-galahad__dgo_8h_1a80425d4671e565a45c13aa026f6897ef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_initialize(data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`dgo_control_type <doxid-structdgo__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The import was succesful.

.. index:: pair: function; dgo_read_specfile
.. _doxid-galahad__dgo_8h_1ab8ba227e6d624a0197afab9f77bbe66a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNDGO.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/dgo.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`dgo_control_type <doxid-structdgo__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; dgo_import
.. _doxid-galahad__dgo_8h_1ace7cbe696d8be7026753681d9b7cd149:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_import(control, data, status, n, x_l, x_u, 
                            H_type, ne, H_row, H_col, H_ptr)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`dgo_control_type <doxid-structdgo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables.

	*
		- x_l

		- is a one-dimensional array of size n and type T that holds the values :math:`x^l` of the lower bounds on the optimization variables :math:`x`. The j-th component of x_l, :math:`j = 0, \ldots, n-1`, contains :math:`x^l_j`.

	*
		- x_u

		- is a one-dimensional array of size n and type T that holds the values :math:`x^u` of the upper bounds on the optimization variables :math:`x`. The j-th component of x_u, :math:`j = 0, \ldots, n-1`, contains :math:`x^u_j`.

	*
		- H_type

		- is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`symmetric storage scheme <doxid-index_1main_symmetric_matrices>` used for the Hessian. It should be one of 'coordinate', 'sparse_by_rows', 'dense', 'diagonal' or 'absent', the latter if access to the Hessian is via matrix-vector products; lower or upper case variants are allowed.

	*
		- ne

		- is a scalar variable of type Int32 that holds the number of entries in the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes.

	*
		- H_row

		- is a one-dimensional array of size ne and type Int32 that holds the row indices of the lower triangular part of H in the sparse co-ordinate storage scheme. It need not be set for any of the other three schemes, and in this case can be NULL

	*
		- H_col

		- is a one-dimensional array of size ne and type Int32 that holds the column indices of the lower triangular part of H in either the sparse co-ordinate, or the sparse row-wise storage scheme. It need not be set when the dense or diagonal storage schemes are used, and in this case can be NULL

	*
		- H_ptr

		- is a one-dimensional array of size n+1 and type Int32 that holds the starting position of each row of the lower triangular part of H, as well as the total number of entries, in the sparse row-wise storage scheme. It need not be set when the other schemes are used, and in this case can be NULL

.. index:: pair: function; dgo_reset_control
.. _doxid-galahad__dgo_8h_1ab52e88675fc811f7e9bc38148d42e932:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_reset_control(control, data, status)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`dgo_control_type <doxid-structdgo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; dgo_solve_with_mat
.. _doxid-galahad__dgo_8h_1a3b573f5a56c7162383a757221a5b7a36:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_solve_with_mat(data, userdata, status, n, x, g, ne, 
                                    eval_f, eval_g, eval_h, eval_hprod, eval_prec)

Find an approximation to the global minimizer of a given function subject to simple bounds on the variables using a partition-and-bound trust-region method.

This call is for the case where :math:`H = \nabla_{xx}f(x)` is provided specifically, and all function/derivative information is available by function calls.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- userdata

		- is a structure that allows data to be passed into the function and derivative evaluation programs.

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -7. The objective function appears to be unbounded from below

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

		  * -91. The hash table used to store the dictionary of vertices of the sub-boxes is full, and there is no room to increase it further.

		  * -99. The budget limit on function evaluations has been reached. This will happen if the limit control.max_evals is exceeded, and is quite normal for stochastic global-optimization methods. The user may explore increasing control.max_evals to see if that produces a lower value of the objective function, but there are unfortunately no guarantees.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- ne

		- is a scalar variable of type Int32 that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- eval_f

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_f( int n, const double x[], double *f, const void *userdata )

		  The value of the objective function :math:`f(x)` evaluated at x= :math:`x` must be assigned to f, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_f`` via the structure ``userdata``.

	*
		- eval_g

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_g( int n, const double x[], double g[], const void *userdata )

		  The components of the gradient :math:`g = \nabla_x f(x`) of the objective function evaluated at x= :math:`x` must be assigned to g, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_g`` via the structure ``userdata``.

	*
		- eval_h

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_h( int n, int ne, const double x[], double h[],
		  	            const void *userdata )

		  The nonzeros of the Hessian :math:`H = \nabla_{xx}f(x)` of the objective function evaluated at x= :math:`x` must be assigned to h in the same order as presented to dgo_import, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_h`` via the structure ``userdata``.

	*
		- eval_prec

		-
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_prec( int n, const double x[], double u[], const double v[],
		  	               const void *userdata )

		  The product :math:`u = P(x) v` of the user's preconditioner :math:`P(x)` evaluated at :math:`x` with the vector v = :math:`v`, the result :math:`u` must be retured in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_prec`` via the structure ``userdata``.

.. index:: pair: function; dgo_solve_without_mat
.. _doxid-galahad__dgo_8h_1a6ea7cfa79c25e784d21e10cc26ed9954:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_solve_without_mat(data, userdata, status, n, x, g, 
                                       eval_f, eval_g, eval_hprod, 
                                       eval_shprod, eval_prec)

Find an approximation to the global minimizer of a given function subject to simple bounds on the variables using a partition-and-bound trust-region method.

This call is for the case where access to :math:`H = \nabla_{xx}f(x)` is provided by Hessian-vector products, and all function/derivative information is available by function calls.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- userdata

		- is a structure that allows data to be passed into the function and derivative evaluation programs.

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -7. The objective function appears to be unbounded from below

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

		  * -99. The budget limit on function evaluations has been reached. This will happen if the limit control.max_evals is exceeded, and is quite normal for stochastic global-optimization methods. The user may explore increasing control.max_evals to see if that produces a lower value of the objective function, but there are unfortunately no guarantees.

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- eval_f

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_f( int n, const double x[], double *f, const void *userdata )

		  The value of the objective function :math:`f(x)` evaluated at x= :math:`x` must be assigned to f, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_f`` via the structure ``userdata``.

	*
		- eval_g

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_g( int n, const double x[], double g[], const void *userdata )

		  The components of the gradient :math:`g = \nabla_x f(x`) of the objective function evaluated at x= :math:`x` must be assigned to g, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_g`` via the structure ``userdata``.

	*
		- eval_hprod

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_hprod( int n, const double x[], double u[], const double v[],
		  	                bool got_h, const void *userdata )

		  The sum :math:`u + \nabla_{xx}f(x) v` of the product of the Hessian :math:`\nabla_{xx}f(x)` of the objective function evaluated at x= :math:`x` with the vector v= :math:`v` and the vector $ :math:`u` must be returned in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. The Hessian has already been evaluated or used at x if got_h is true. Data may be passed into ``eval_hprod`` via the structure ``userdata``.

	*
		- eval_shprod

		-
		  is a user-supplied function that must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_shprod( int n, const double x[], int nnz_v,
		  	                 const int index_nz_v[], const double v[],
		  	                 int *nnz_u, int index_nz_u[], double u[],
		  	                 bool got_h, const void *userdata )

		  The product :math:`u = \nabla_{xx}f(x) v` of the Hessian :math:`\nabla_{xx}f(x)` of the objective function evaluated at :math:`x` with the sparse vector v= :math:`v` must be returned in u, and the function return value set to 0. Only the components index_nz_v[0:nnz_v-1] of v are nonzero, and the remaining components may not have been be set. On exit, the user must indicate the nnz_u indices of u that are nonzero in index_nz_u[0:nnz_u-1], and only these components of u need be set. If the evaluation is impossible at x, return should be set to a nonzero value. The Hessian has already been evaluated or used at x if got_h is true. Data may be passed into ``eval_prec`` via the structure ``userdata``.

	*
		- eval_prec

		-
		  is an optional user-supplied function that may be NULL. If non-NULL, it must have the following signature:

		  .. ref-code-block:: julia

		  	Int32 eval_prec( int n, const double x[], double u[], const double v[],
		  	               const void *userdata )

		  The product :math:`u = P(x) v` of the user's preconditioner :math:`P(x)` evaluated at :math:`x` with the vector v = :math:`v`, the result :math:`u` must be retured in u, and the function return value set to 0. If the evaluation is impossible at x, return should be set to a nonzero value. Data may be passed into ``eval_prec`` via the structure ``userdata``.

.. index:: pair: function; dgo_solve_reverse_with_mat
.. _doxid-galahad__dgo_8h_1a02f408b215596c01b0e3836dfa301b9f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_solve_reverse_with_mat(data, status, eval_status, 
                                            n, x, f, g, ne, H_val, u, v)

Find an approximation to the global minimizer of a given function subject to simple bounds on the variables using a partition-and-bound trust-region method.

This call is for the case where :math:`H = \nabla_{xx}f(x)` is provided specifically, but function/derivative information is only available by returning to the calling procedure



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -7. The objective function appears to be unbounded from below

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

		  * -99. The budget limit on function evaluations has been reached. This will happen if the limit control.max_evals is exceeded, and is quite normal for stochastic global-optimization methods. The user may explore increasing control.max_evals to see if that produces a lower value of the objective function, but there are unfortunately no guarantees.

		  * 2. The user should compute the objective function value :math:`f(x)` at the point :math:`x` indicated in x and then re-enter the function. The required value should be set in f, and eval_status should be set to 0. If the user is unable to evaluate :math:`f(x)` for instance, if the function is undefined at :math:`x` the user need not set f, but should then set eval_status to a non-zero value.

		  * 3. The user should compute the gradient of the objective function :math:`\nabla_x f(x)` at the point :math:`x` indicated in x and then re-enter the function. The value of the i-th component of the g radient should be set in g[i], for i = 0, ..., n-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`\nabla_x f(x)` for instance if a component of the gradient is undefined at :math:`x` -the user need not set g, but should then set eval_status to a non-zero value.

		  * 4. The user should compute the Hessian of the objective function :math:`\nabla_{xx}f(x)` at the point x indicated in :math:`x` and then re-enter the function. The value l-th component of the Hessian stored according to the scheme input in the remainder of :math:`H` should be set in H_val[l], for l = 0, ..., ne-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`\nabla_{xx}f(x)` for instance, if a component of the Hessian is undefined at :math:`x` the user need not set H_val, but should then set eval_status to a non-zero value.

		  * 5. The user should compute the product :math:`\nabla_{xx}f(x)v` of the Hessian of the objective function :math:`\nabla_{xx}f(x)` at the point :math:`x` indicated in x with the vector :math:`v`, add the result to the vector :math:`u` and then re-enter the function. The vectors :math:`u` and :math:`v` are given in u and v respectively, the resulting vector :math:`u + \nabla_{xx}f(x)v` should be set in u and eval_status should be set to 0. If the user is unable to evaluate the product for instance, if a component of the Hessian is undefined at :math:`x` the user need not alter u, but should then set eval_status to a non-zero value.

		  * 6. The user should compute the product :math:`u = P(x)v` of their preconditioner :math:`P(x)` at the point x indicated in :math:`x` with the vector :math:`v` and then re-enter the function. The vector :math:`v` is given in v, the resulting vector :math:`u = P(x)v` should be set in u and eval_status should be set to 0. If the user is unable to evaluate the product for instance, if a component of the preconditioner is undefined at :math:`x` the user need not set u, but should then set eval_status to a non-zero value.

		  * 23. The user should follow the instructions for 2 **and** 3 above before returning.

		  * 25. The user should follow the instructions for 2 **and** 5 above before returning.

		  * 35. The user should follow the instructions for 3 **and** 5 above before returning.

		  * 235. The user should follow the instructions for 2, 3 **and** 5 above before returning.

	*
		- eval_status

		- is a scalar variable of type Int32 that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- f

		- is a scalar variable pointer of type T that holds the value of the objective function.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- ne

		- is a scalar variable of type Int32 that holds the number of entries in the lower triangular part of the Hessian matrix :math:`H`.

	*
		- H_val

		- is a one-dimensional array of size ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix :math:`H` in any of the available storage schemes.

	*
		- u

		- is a one-dimensional array of size n and type T that is used for reverse communication (see above for details)

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication (see above for details)

.. index:: pair: function; dgo_solve_reverse_without_mat
.. _doxid-galahad__dgo_8h_1a878a7d98d55794fa38f885a5d76aa4f0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_solve_reverse_without_mat(data, status, eval_status, 
                                                n, x, f, g, u, v, index_nz_v, 
                                                nnz_v, index_nz_u, nnz_u)

Find an approximation to the global minimizer of a given function subject to simple bounds on the variables using a partition-and-bound trust-region method.

This call is for the case where access to :math:`H = \nabla_{xx}f(x)` is provided by Hessian-vector products, but function/derivative information is only available by returning to the calling procedure.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the entry and exit status from the package.

		  On initial entry, status must be set to 1.

		  Possible exit are:

		  * 0. The run was succesful



		  * -1. An allocation error occurred. A message indicating the offending array is written on unit control.error, and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -2. A deallocation error occurred. A message indicating the offending array is written on unit control.error and the returned allocation status and a string containing the name of the offending array are held in inform.alloc_status and inform.bad_alloc respectively.

		  * -3. The restriction n > 0 or requirement that type contains its relevant string 'dense', 'coordinate', 'sparse_by_rows', 'diagonal' or 'absent' has been violated.

		  * -7. The objective function appears to be unbounded from below

		  * -9. The analysis phase of the factorization failed; the return status from the factorization package is given in the component inform.factor_status

		  * -10. The factorization failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -11. The solution of a set of linear equations using factors from the factorization package failed; the return status from the factorization package is given in the component inform.factor_status.

		  * -16. The problem is so ill-conditioned that further progress is impossible.

		  * -19. The CPU time limit has been reached. This may happen if control.cpu_time_limit is too small, but may also be symptomatic of a badly scaled problem.

		  * -82. The user has forced termination of solver by removing the file named control.alive_file from unit unit control.alive_unit.

		  * -99. The budget limit on function evaluations has been reached. This will happen if the limit control.max_evals is exceeded, and is quite normal for stochastic global-optimization methods. The user may explore increasing control.max_evals to see if that produces a lower value of the objective function, but there are unfortunately no guarantees.

		  * 2. The user should compute the objective function value :math:`f(x)` at the point :math:`x` indicated in x and then re-enter the function. The required value should be set in f, and eval_status should be set to 0. If the user is unable to evaluate :math:`f(x)` for instance, if the function is undefined at :math:`x` the user need not set f, but should then set eval_status to a non-zero value.

		  * 3. The user should compute the gradient of the objective function :math:`\nabla_x f(x)` at the point :math:`x` indicated in x and then re-enter the function. The value of the i-th component of the g radient should be set in g[i], for i = 0, ..., n-1 and eval_status should be set to 0. If the user is unable to evaluate a component of :math:`\nabla_x f(x)` for instance if a component of the gradient is undefined at :math:`x` -the user need not set g, but should then set eval_status to a non-zero value.

		  * 5. The user should compute the product :math:`\nabla_{xx}f(x)v` of the Hessian of the objective function :math:`\nabla_{xx}f(x)` at the point :math:`x` indicated in x with the vector :math:`v`, add the result to the vector :math:`u` and then re-enter the function. The vectors :math:`u` and :math:`v` are given in u and v respectively, the resulting vector :math:`u + \nabla_{xx}f(x)v` should be set in u and eval_status should be set to 0. If the user is unable to evaluate the product for instance, if a component of the Hessian is undefined at :math:`x` the user need not alter u, but should then set eval_status to a non-zero value.

		  * 6. The user should compute the product :math:`u = P(x)v` of their preconditioner :math:`P(x)` at the point x indicated in :math:`x` with the vector :math:`v` and then re-enter the function. The vector :math:`v` is given in v, the resulting vector :math:`u = P(x)v` should be set in u and eval_status should be set to 0. If the user is unable to evaluate the product for instance, if a component of the preconditioner is undefined at :math:`x` the user need not set u, but should then set eval_status to a non-zero value.

		  * 7. The user should compute the product :math:`u = \nabla_{xx}f(x)v` of the Hessian of the objective function :math:`\nabla_{xx}f(x)` at the point :math:`x` indicated in x with the **sparse** vector v= :math:`v` and then re-enter the function. The nonzeros of :math:`v` are stored in v[index_nz_v[0:nnz_v-1]] while the nonzeros of :math:`u` should be returned in u[index_nz_u[0:nnz_u-1]]; the user must set nnz_u and index_nz_u accordingly, and set eval_status to 0. If the user is unable to evaluate the product for instance, if a component of the Hessian is undefined at :math:`x` the user need not alter u, but should then set eval_status to a non-zero value.

		  * 23. The user should follow the instructions for 2 **and** 3 above before returning.

		  * 25. The user should follow the instructions for 2 **and** 5 above before returning.

		  * 35. The user should follow the instructions for 3 **and** 5 above before returning.

		  * 235. The user should follow the instructions for 2, 3 **and** 5 above before returning.

	*
		- eval_status

		- is a scalar variable of type Int32 that is used to indicate if objective function/gradient/Hessian values can be provided (see above)

	*
		- n

		- is a scalar variable of type Int32 that holds the number of variables

	*
		- x

		- is a one-dimensional array of size n and type T that holds the values :math:`x` of the optimization variables. The j-th component of x, j = 0, ... , n-1, contains :math:`x_j`.

	*
		- f

		- is a scalar variable pointer of type T that holds the value of the objective function.

	*
		- g

		- is a one-dimensional array of size n and type T that holds the gradient :math:`g = \nabla_xf(x)` of the objective function. The j-th component of g, j = 0, ... , n-1, contains :math:`g_j`.

	*
		- u

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=5,6,7 above for details)

	*
		- v

		- is a one-dimensional array of size n and type T that is used for reverse communication (see status=5,6,7 above for details)

	*
		- index_nz_v

		- is a one-dimensional array of size n and type Int32 that is used for reverse communication (see status=7 above for details)

	*
		- nnz_v

		- is a scalar variable of type Int32 that is used for reverse communication (see status=7 above for details)

	*
		- index_nz_u

		- s a one-dimensional array of size n and type Int32 that is used for reverse communication (see status=7 above for details)

	*
		- nnz_u

		- is a scalar variable of type Int32 that is used for reverse communication (see status=7 above for details). On initial (status=1) entry, nnz_u should be set to an (arbitrary) nonzero value, and nnz_u=0 is recommended

.. index:: pair: function; dgo_information
.. _doxid-galahad__dgo_8h_1aea0c208de08f507be7a31fe3ab7d3b91:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_information(data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`dgo_inform_type <doxid-structdgo__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; dgo_terminate
.. _doxid-galahad__dgo_8h_1ad12337a0c7ad3ac74e7f8c0783fbbfab:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function dgo_terminate(data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`dgo_control_type <doxid-structdgo__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`dgo_inform_type <doxid-structdgo__inform__type>`)
