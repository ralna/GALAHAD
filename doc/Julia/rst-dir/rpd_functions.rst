callable functions
------------------

.. index:: pair: function; rpd_initialize
.. _doxid-galahad__rpd_8h_1a6805ebb5cc097db7df39723c64cef793:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`rpd_control_type <doxid-structrpd__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; rpd_get_stats
.. _doxid-galahad__rpd_8h_1ad0148374adcd7bf5f34f378ba0995a21:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_stats(T, INT, qplib_file, qplib_file_len, control, data, 
                               status, p_type, n, m, h_ne, a_ne, h_c_ne)

Read the data from a specified QPLIB file into internal storage, and report the type of problem encoded, along with problem-specific dimensions.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- qplib_file

		- is a one-dimensional array of type Vararg{Cchar} that specifies the name of the QPLIB file that is to be read.

	*
		- qplib_file_len

		- is a scalar variable of type INT that gives the number of characters in the name encoded in qplib_file.

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`rpd_control_type <doxid-structrpd__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-1**
                    An allocation error occurred. A message indicating
                    the offending array is written on unit
                    control.error, and the returned allocation status
                    and a string containing the name of the offending
                    array are held in inform.alloc_status and
                    inform.bad_alloc respectively.

		  * **-2**
                    A deallocation error occurred. A message indicating
                    the offending array is written on unit control.error
                    and the returned allocation status and a string
                    containing the name of the offending array are held
                    in inform.alloc_status and inform.bad_alloc
                    respectively.

	*
		- p_type

		-
		  is a one-dimensional array of size 4 and type Vararg{Cchar} that specifies the type of quadratic programming problem encoded in the QPLIB file.



		  The first character indicates the type of objective function used. It will be one of the following:

		  * L a linear objective function.

		  * D a convex quadratic objective function whose Hessian is a diagonal matrix.

		  * C a convex quadratic objective function.

		  * Q a quadratic objective function whose Hessian may be indefinite.



		  The second character indicates the types of variables that are present. It will be one of the following:

		  * C all the variables are continuous.

		  * B all the variables are binary (0-1).

		  * M the variables are a mix of continuous and binary.

		  * I all the variables are integer.

		  * G the variables are a mix of continuous, binary and integer.



		  The third character indicates the type of the (most extreme) constraint function used; other constraints may be of a lesser type. It will be one of the following:

		  * N there are no constraints.

		  * B some of the variables lie between lower and upper bounds (box constraint).

		  * L the constraint functions are linear.

		  * D the constraint functions are convex quadratics with diagonal Hessians.

		  * C the constraint functions are convex quadratics.

		  * Q the constraint functions are quadratics whose Hessians may be indefinite.

		  Thus for continuous problems, we would have

		  * LCL a linear program.

		  * LCC or LCQ a linear program with quadratic constraints.

		  * CCB or QCB a bound-constrained quadratic program.

		  * CCL or QCL a quadratic program.

		  * CCC or CCQ or QCC or QCQ a quadratic program with quadratic constraints.

		  For integer problems, the second character would be I rather than C, and for mixed integer problems, the second character would by M or G.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- m

		- is a scalar variable of type INT that holds the number of general constraints.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H$ stored in the sparse symmetric co-ordinate storage scheme.

	*
		- a_ne

		- is a scalar variable of type INT that holds the number of entries in $A$ stored in the sparse co-ordinate storage scheme.

	*
		- h_c_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of $H_c$ stored in the joint sparse co-ordinate storage scheme.

.. index:: pair: function; rpd_get_g
.. _doxid-galahad__rpd_8h_1aa5be687c00e4a7980c5ea7c258717d3a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_g(T, INT, data, status, n, g)

Recover the linear term $g$ from in objective function

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- g

		- is a one-dimensional array of size n and type T that gives the linear term $g$ of the objective function. The j-th component of ``g``, j = 1, ... , n, contains $g_j$.

.. index:: pair: function; rpd_get_f
.. _doxid-galahad__rpd_8h_1a38dc68ed79b192e3fcd961b8589d202c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_f(T, INT, data, status, f)

Recover the constant term $f$ in the objective function.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- f

		- is a scalar of type T that gives the constant term $f$ from the objective function.

.. index:: pair: function; rpd_get_xlu
.. _doxid-galahad__rpd_8h_1a6a5cbf68b561cc6db0ba08304d28787c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_xlu(T, INT, data, status, n, x_l, x_u)

Recover the variable lower and upper bounds $x_l$ and $x_u$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- x_l

		- is a one-dimensional array of size n and type T that gives the lower bounds $x_l$ on the variables $x$. The j-th component of ``x_l``, j = 1, ... , n, contains $(x_l)_j$.

	*
		- x_u

		- is a one-dimensional array of size n and type T that gives the upper bounds $x_u$ on the variables $x$. The j-th component of ``x_u``, j = 1, ... , n, contains $(x_u)_j$.

.. index:: pair: function; rpd_get_clu
.. _doxid-galahad__rpd_8h_1aa3b44968b109ed194ed2bb04009f35ac:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_clu(T, INT, data, status, m, c_l, c_u)

Recover the constraint lower and upper bounds $c_l$ and $c_u$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- m

		- is a scalar variable of type INT that holds the number of general constraints.

	*
		- c_l

		- is a one-dimensional array of size m and type T that gives the lower bounds $c_l$ on the constraints $A x$. The i-th component of ``c_l``, i = 1, ... , m, contains $(c_l)_i$.

	*
		- c_u

		- is a one-dimensional array of size m and type T that gives the upper bounds $c_u$ on the constraints $A x$. The i-th component of ``c_u``, i = 1, ... , m, contains $(c_u)_i$.

.. index:: pair: function; rpd_get_h
.. _doxid-galahad__rpd_8h_1a02021324df6f485160d327f2f5fca0d3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_h(T, INT, data, status, h_ne, h_row, h_col, h_val)

Recover the Hessian term $H$ in the objective function.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- h_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- h_row

		- is a one-dimensional array of size h_ne and type INT that gives the row indices of the lower triangular part of $H$ in the :ref:`sparse co-ordinate storage scheme<details-rpd_storage>`.

	*
		- h_col

		- is a one-dimensional array of size h_ne and type INT that gives the column indices of the lower triangular part of $H$ in the sparse co-ordinate storage scheme.

	*
		- h_val

		- is a one-dimensional array of size h_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H$ in the sparse co-ordinate storage scheme.

.. index:: pair: function; rpd_get_a
.. _doxid-galahad__rpd_8h_1a8b0c3c507b12512b09ee4ec92596148e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_a(T, INT, data, status, a_ne, a_row, a_col, a_val)

Recover the Jacobian term $A$ in the constraints.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- a_ne

		- is a scalar variable of type INT that holds the number of entries in the constraint Jacobian matrix $A$.

	*
		- a_row

		- is a one-dimensional array of size a_ne and type INT that gives the row indices of $A$ in the :ref:`sparse co-ordinate storage scheme<details-rpd_storage>`.

	*
		- a_col

		- is a one-dimensional array of size a_ne and type INT that gives the column indices of $A$ in the sparse co-ordinate, storage scheme.

	*
		- a_val

		- is a one-dimensional array of size a_ne and type T that gives the values of the entries of the constraint Jacobian matrix $A$ in the sparse co-ordinate scheme.

.. index:: pair: function; rpd_get_h_c
.. _doxid-galahad__rpd_8h_1a55ae091188ad0d88920565549bd47451:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_h_c(T, INT, data, status, h_c_ne, 
                             h_c_ptr, h_c_row, h_c_col, h_c_val)

Recover the Hessian terms $H_c$ in the constraints.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- h_c_ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the Hessian matrix $H$.

	*
		- h_c_ptr

		- is a one-dimensional array of size h_c_ne and type INT that gives the constraint indices of the lower triangular part of $H_c$ in the :ref:`joint sparse co-ordinate storage scheme <details-rpd_storage>`.

	*
		- h_c_row

		- is a one-dimensional array of size h_c_ne and type INT that gives the row indices of the lower triangular part of $H_c$ in the joint sparse co-ordinate storage scheme.

	*
		- h_c_col

		- is a one-dimensional array of size h_c_ne and type INT that gives the column indices of the lower triangular part of $H_c$ in the sparse co-ordinate storage scheme.

	*
		- h_c_val

		- is a one-dimensional array of size h_c_ne and type T that holds the values of the entries of the lower triangular part of the Hessian matrix $H_c$ in the sparse co-ordinate storage scheme.

.. index:: pair: function; rpd_get_x_type
.. _doxid-galahad__rpd_8h_1af784ecc65c925575788a494bd8118f4d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_x_type(T, INT, data, status, n, x_type)

Recover the types of the variables $x$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- x_type

		-
		  is a one-dimensional array of size n and type INT that specifies the type of each variable $x$. Specifically, for j = 1, ... , n, x(j) =

		  * 0 if variable $x_j$ is continuous,

		  * 1 if variable $x_j$ is integer, and

		  * 2 if variable $x_j$ is binary (0,1)

.. index:: pair: function; rpd_get_x
.. _doxid-galahad__rpd_8h_1afbc831595295e9153e4740d852a35c27:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_x(T, INT, data, status, n,

Recover the initial values of the variables $x$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- x

		- is a one-dimensional array of size n and type T that gives the initial values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

.. index:: pair: function; rpd_get_y
.. _doxid-galahad__rpd_8h_1ac9fd1a08acf460b7962ad5393d69fff5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_y(T, INT, data, status, m, y)

Recover the initial values of the Lagrange multipliers $y$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- m

		- is a scalar variable of type INT that holds the number of general constraints.

	*
		- y

		- is a one-dimensional array of size n and type T that gives the initial values $y$ of the Lagrange multipliers for the general constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

.. index:: pair: function; rpd_get_z
.. _doxid-galahad__rpd_8h_1ab1579a81766096bd1764f0fb0cc10db3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_get_z(T, INT, data, status, n, z)

Recover the initial values of the dual variables $z$.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The statistics have been recovered successfully.

		  * **-93**
                    The QPLIB file did not contain the required data.

	*
		- n

		- is a scalar variable of type INT that holds the number of variables.

	*
		- z

		- is a one-dimensional array of size n and type T that gives the initial values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

.. index:: pair: function; rpd_information
.. _doxid-galahad__rpd_8h_1a6deb3fc67d1b4e1d1cd1661af237d6b3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`rpd_inform_type <doxid-structrpd__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; rpd_terminate
.. _doxid-galahad__rpd_8h_1af49fc46839c605dd71d2666189d0d8a9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function rpd_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`rpd_control_type <doxid-structrpd__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`rpd_inform_type <doxid-structrpd__inform__type>`)
