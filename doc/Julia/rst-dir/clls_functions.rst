callable functions
------------------

.. index:: pair: function; clls_initialize
.. _doxid-galahad__clls_8h_1a782387ad9cccc5f2e2da9df9016fb923:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function clls_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The import was successful.

.. index:: pair: function; clls_read_specfile
.. _doxid-galahad__clls_8h_1ade439e5e06c2852fcb089bb39a667a74:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function clls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.
An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list of keywords 
with associated default values is provided in \$GALAHAD/src/clls/CLLS.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/clls.pdf for a list of how these keywords relate to the 
components of the control structure.

.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - control

                - is a structure containing control information (see :ref:`clls_control_type <doxid-structclls__control__type>`)

        *
                - specfile

                - is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; clls_import
.. _doxid-galahad__clls_8h_1a6a2be17b6f871df80bbac93940b83af3:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function clls_import(T, INT, control, data, status, n, o, m, 
                             Ao_type, Ao_ne, Ao_row, Ao_col, Ao_ptr_ne, Ao_ptr, 
                             A_type, A_ne, A_row, A_col, A_ptr_ne, A_ptr)

Import problem data into internal storage prior to solution.


.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - control

                - is a structure whose members provide control parameters for the remaining procedures (see :ref:`clls_control_type <doxid-structclls__control__type>`)

        *
                - data

                - holds private internal data

        *
                - status

                - is a scalar variable of type INT that gives the exit
                  status from the package. Possible values are:

                  * **0**
                    The import was successful

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

                  * **-3**
                    The restrictions n > 0, o > 0 or m $\geq$ 0 or requirement 
                    that a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.

        *
                - n

                - is a scalar variable of type INT that holds the number of variables.

        *
                - o

                - is a scalar variable of type INT that holds the number of residuals.

        *
                - m

                - is a scalar variable of type INT that holds the number of general linear constraints.

        *
                - Ao_type

                - is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the objective design matrix, $A_o$. It should be one of 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense' or 'dense_by_columns'; lower or upper case variants are allowed.

        *
                - Ao_ne

                - is a scalar variable of type INT that holds the number of entries in $A_o$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

        *
                - Ao_row

                - is a one-dimensional array of size Ao_ne and type INT that holds the row indices of $A_o$ in the sparse co-ordinate and sparse column-wise storage schemes. It need not be set for any of the other schemes, and in this case can be C_NULL.

        *
                - Ao_col

                - is a one-dimensional array of size Ao_ne and type INT that holds the column indices of $A_o$ in the sparse co-ordinate and the sparse row-wise storage schemes. It need not be set for any of the other schemes, and in this case can be C_NULL.

        *
                - Ao_ptr_ne

                - is a scalar variable of type INT that holds the length of the pointer array if sparse row or column storage scheme is used for $A_o$. For the sparse row scheme,  Ao_ptr_ne should be at least o+1, while for the sparse column scheme,  it should be at least n+1, It need not be set when the other schemes are used.

        *
                - Ao_ptr

                - is a one-dimensional array of size n+1 and type INT that holds the starting position of each row of $A_o$, as well as the total number of entries, in the sparse row-wise storage scheme. By contrast, it is a one-dimensional array of size n+1 and type INT that holds the starting position of each column of $A_o$, as well as the total number of entries, in the sparse column-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.

        *
                - A_type

                - is a one-dimensional array of type Vararg{Cchar} that specifies the :ref:`unsymmetric storage scheme <doxid-index_1main_unsymmetric_matrices>` used for the constraint Jacobian, $A$. It should be one of 'coordinate', 'sparse_by_rows', 'sparse_by_columns', 'dense' or 'dense_by_columns'; lower or upper case variants are allowed.

        *
                - A_ne

                - is a scalar variable of type INT that holds the number of entries in $A$ in the sparse co-ordinate storage scheme. It need not be set for any of the other schemes.

        *
                - A_row

                - is a one-dimensional array of size A_ne and type INT that holds the row indices of $A$ in the sparse co-ordinate and sparse column-wise storage schemes. It need not be set for any of the other schemes, and in this case can be C_NULL.

        *
                - A_col

                - is a one-dimensional array of size A_ne and type INT that holds the column indices of $A$ in the sparse co-ordinate and the sparse row-wise storage schemes. It need not be set for any of the other schemes, and in this case can be C_NULL.

        *
                - A_ptr_ne

                - is a scalar variable of type INT that holds the length of the pointer array if sparse row or column storage scheme is used for $A$. For the sparse row scheme,  A_ptr_ne should be at least o+1, while for the sparse column scheme,  it should be at least n+1, It need not be set when the other schemes are used.

        *
                - A_ptr

                - is a one-dimensional array of size n+1 and type INT 
that holds the starting position of each row of $A$, as well as the total number of entries, in the sparse row-wise storage scheme. By contrast, it is a one-dimensional array of size n+1 and type INT that holds the starting position of each column of $A$, as well as the total number of entries, in the sparse column-wise storage scheme. It need not be set when the other schemes are used, and in this case can be C_NULL.




.. index:: pair: function; clls_reset_control
.. _doxid-galahad__clls_8h_1a9f7ccb0cffa909a2be7556edda430190:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function clls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - control

                - is a structure whose members provide control parameters for the remaining procedures (see :ref:`clls_control_type <doxid-structclls__control__type>`)

        *
                - data

                - holds private internal data

        *
                - status

                - is a scalar variable of type INT that gives the exit
                  status from the package. Possible values are:

                  * **0**
                    The import was successful.

.. index:: pair: function; clls_solve_clls
.. _doxid-galahad__clls_8h_1ac2d720ee7b719bf63c3fa208d37f1bc1:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function clls_solve_clls(T, INT, data, status, n, o, m, 
                                 Ao_ne, Ao_val, b, sigma, a_ne, A_val, 
                                 c_l, c_u, x_l, x_u, x, r, c, y, z, 
                                 x_stat, c_stat, w)

Solve the linearly-constrained regularized linear least-squares problem.

.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - data

                - holds private internal data

        *
                - status

                - is a scalar variable of type INT that gives the
                  entry and exit status from the package.

                  Possible exit values are:

                  * **0**
                    The run was successful.

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

                  * **-3**
                    The restrictions n > 0, o > 0 and m $\geq$ 0 or requirement
                    that a type contains its relevant string 'dense',
                    'coordinate', 'sparse_by_rows', 'diagonal',
                    'scaled_identity', 'identity', 'zero' or 'none' has
                    been violated.

                  * **-5**
                    The simple-bound constraints are inconsistent.

                  * **-7**
                    The constraints appear to have no feasible point.

                  * **-9**
                    The analysis phase of the factorization failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status

                  * **-10**
                    The factorization failed; the return status from the
                    factorization package is given in the component
                    inform.factor_status.

                  * **-11**
                    The solution of a set of linear equations using
                    factors from the factorization package failed; the
                    return status from the factorization package is
                    given in the component inform.factor_status.

                  * **-16**
                    The problem is so ill-conditioned that further
                    progress is impossible.

                  * **-17**
                    The step is too small to make further impact.

                  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.

                  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.

        *
                - n

                - is a scalar variable of type INT that holds the number of variables

        *
                - o

                - is a scalar variable of type INT that holds the number of residulas.

        *
                - m

                - is a scalar variable of type INT that holds the number of general linear constraints.

        *
                - Ao_ne

                - is a scalar variable of type INT that holds the number of entries in the objectve design matrix $A_o$.

        *
                - Ao_val

                - is a one-dimensional array of size Ao_ne and type T that holds the values of the entries of the design matrix $A_o$ in any of the available storage schemes.

        *
                - b

                - is a one-dimensional array of size o and type T that holds the observations $b$. The j-th component of ``b``, i = 1, ... , o, contains $b_i$.

        *
                - sigma

                - is a scalar of type T that holds the non-negative regularization weight $\sigma \geq 0$.

        *
                - a_ne

                - is a scalar variable of type INT that holds the number of entries in the constraint Jacobian matrix $A$.

        *
                - A_val

                - is a one-dimensional array of size a_ne and type T that holds the values of the entries of the constraint Jacobian matrix $A$ in any of the available storage schemes.

        *
                - c_l

                - is a one-dimensional array of size m and type T that holds the lower bounds $c^l$ on the constraints $A x$. The i-th component of ``c_l``, i = 1, ... , m, contains $c^l_i$.

        *
                - c_u

                - is a one-dimensional array of size m and type T that holds the upper bounds $c^l$ on the constraints $A x$. The i-th component of ``c_u``, i = 1, ... , m, contains $c^u_i$.

        *
                - x_l

                - is a one-dimensional array of size n and type T that holds the lower bounds $x^l$ on the variables $x$. The j-th component of ``x_l``, j = 1, ... , n, contains $x^l_j$.

        *
                - x_u

                - is a one-dimensional array of size n and type T that holds the upper bounds $x^l$ on the variables $x$. The j-th component of ``x_u``, j = 1, ... , n, contains $x^l_j$.

        *
                - x

                - is a one-dimensional array of size n and type T that holds the values $x$ of the optimization variables. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

        *
                - r

                - is a one-dimensional array of size o and type T that holds the residual $r(x) = A_o x - b$. The i-th component of ``b``, i = 1, ... , o, contains $r_i(x)$.

        *
                - c

                - is a one-dimensional array of size m and type T that holds the constraint residual $c(x) = A x$. The i-th component of ``c``, i = 1, ... , m, contains $c_i(x)$.

        *
                - y

                - is a one-dimensional array of size n and type T that holds the values $y$ of the Lagrange multipliers for the general linear constraints. The j-th component of ``y``, j = 1, ... , m, contains $y_j$.

        *
                - z

                - is a one-dimensional array of size n and type T that holds the values $z$ of the dual variables. The j-th component of ``z``, j = 1, ... , n, contains $z_j$.

        *
                - x_stat

                - is a one-dimensional array of size n and type INT that gives the optimal status of the problem variables. If x_stat(j) is negative, the variable $x_j$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

        *
                - c_stat

                - is a one-dimensional array of size m and type INT that gives the optimal status of the general linear constraints. If c_stat(i) is negative, the constraint value $a_i^Tx$ most likely lies on its lower bound, if it is positive, it lies on its upper bound, and if it is zero, it lies between its bounds.

        *
                - w

                - is a one-dimensional array of size o and type T that holds the values $w$ of strictly-positive observation weights. The i-th component of w, i = 0, ... , o-1, contains $w_i$. If the weights are all one, w can be set to C_NULL.


.. index:: pair: function; clls_information
.. _doxid-galahad__clls_8h_1adfb7589696e4e07fdb65f02bc42c5daf:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function clls_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
        :widths: 20 80

        *
                - data

                - holds private internal data

        *
                - inform

                - is a structure containing output information (see :ref:`clls_inform_type <doxid-structclls__inform__type>`)

        *
                - status

                - is a scalar variable of type INT that gives the exit
                  status from the package. Possible values are
                  (currently):

                  * **0**
                    The values were recorded successfully

.. index:: pair: function; clls_terminate
.. _doxid-galahad__clls_8h_1a84e12e9e546f51762d305333dce68e2b:

.. ref-code-block:: julia
        :class: doxyrest-title-code-block

        function clls_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`clls_control_type <doxid-structclls__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`clls_inform_type <doxid-structclls__inform__type>`)
