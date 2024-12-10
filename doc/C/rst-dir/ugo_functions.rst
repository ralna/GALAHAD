.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_ugo_control_type.rst
	struct_ugo_inform_type.rst
	struct_ugo_time_type.rst


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	

	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`ugo_control_type<doxid-structugo__control__type>`;
	struct :ref:`ugo_inform_type<doxid-structugo__inform__type>`;
	struct :ref:`ugo_time_type<doxid-structugo__time__type>`;

	// function calls

	void :ref:`ugo_initialize<doxid-galahad__ugo_8h_1a172105bd528410f7c7e2fd77899ebc78>`(void **data, struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);
	void :ref:`ugo_read_specfile<doxid-galahad__ugo_8h_1a6819d58a728f3bf97232ed719e72fb91>`(struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control, const char specfile[]);

	void :ref:`ugo_import<doxid-galahad__ugo_8h_1a8bcbdf9ef1229535b77d9991eb543dcb>`(
		struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>`* x_l,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>`* x_u
	);

	void :ref:`ugo_reset_control<doxid-galahad__ugo_8h_1a51fa6faacfb75c3dcad44befd2e6cb40>`(
		struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	);

	void :ref:`ugo_solve_direct<doxid-galahad__ugo_8h_1aa5b2949ab17e25a0a0c24f38c0d61a1a>`(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* x,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* g,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* h,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`rpc_<doxid-galahad__rpc_8h_>`, :ref:`rpc_<doxid-galahad__rpc_8h_>`*, :ref:`rpc_<doxid-galahad__rpc_8h_>`*, :ref:`rpc_<doxid-galahad__rpc_8h_>`*, const void*) eval_fgh
	);

	void :ref:`ugo_solve_reverse<doxid-galahad__ugo_8h_1a0b8f123f8e67bb0cb8a27c5ce87c824c>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* x,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* g,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* h
	);

	void :ref:`ugo_information<doxid-galahad__ugo_8h_1a8e1db35daea3247b2cc9eb8607d0abee>`(void **data, struct :ref:`ugo_inform_type<doxid-structugo__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`ugo_terminate<doxid-galahad__ugo_8h_1ad9485926c547bb783aea3ee1adb3b084>`(
		void **data,
		struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control,
		struct :ref:`ugo_inform_type<doxid-structugo__inform__type>`* inform
	);

.. _details-global:

typedefs
--------

.. index:: pair: typedef; spc_
.. _doxid-galahad__spc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef float spc_

``spc_`` is real single precision

.. index:: pair: typedef; rpc_
.. _doxid-galahad__rpc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef double rpc_

``rpc_`` is the real working precision used, but may be changed to ``float`` by
defining the  preprocessor variable ``REAL_32`` or (if supported) to
``__real128`` using the variable ``REAL_128``.

.. index:: pair: typedef; ipc_
.. _doxid-galahad__ipc_8h_:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef int ipc_

``ipc_`` is the default integer word length used, but may be changed to 
``int64_t`` by defining the  preprocessor variable ``INTEGER_64``.

function calls
--------------

.. index:: pair: function; ugo_initialize
.. _doxid-galahad__ugo_8h_1a172105bd528410f7c7e2fd77899ebc78:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_initialize(void **data, struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The initialization was successful.

.. index:: pair: function; ugo_read_specfile
.. _doxid-galahad__ugo_8h_1a6819d58a728f3bf97232ed719e72fb91:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_read_specfile(struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is 
:ref:`available<details-spec_file>`, and a detailed list 
of keywords with associated default values is provided in 
\$GALAHAD/src/ugo/UGO.template. 
See also Table 2.1 in the Fortran documentation provided in 
\$GALAHAD/doc/ugo.pdf for a list of how these keywords 
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct containing control information (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; ugo_import
.. _doxid-galahad__ugo_8h_1a8bcbdf9ef1229535b77d9991eb543dcb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_import(
		struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>`* x_l,
		const :ref:`rpc_<doxid-galahad__rpc_8h_>`* x_u
	)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase
		  
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
		- x_l

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value $x^l$ of the lower bound on the optimization variable $x$.

	*
		- x_u

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value $x^u$ of the upper bound on the optimization variable $x$.

.. index:: pair: function; ugo_reset_control
.. _doxid-galahad__ugo_8h_1a51fa6faacfb75c3dcad44befd2e6cb40:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_reset_control(
		struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status
	)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a struct whose members provide control paramters for the remaining prcedures (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are:
		  
		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; ugo_solve_direct
.. _doxid-galahad__ugo_8h_1aa5b2949ab17e25a0a0c24f38c0d61a1a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_solve_direct(
		void **data,
		void *userdata,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* x,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* g,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* h,
		:ref:`ipc_<doxid-galahad__ipc_8h_>`(*)(:ref:`rpc_<doxid-galahad__rpc_8h_>`, :ref:`rpc_<doxid-galahad__rpc_8h_>`*, :ref:`rpc_<doxid-galahad__rpc_8h_>`*, :ref:`rpc_<doxid-galahad__rpc_8h_>`*, const void*) eval_fgh
	)

Find an approximation to the global minimizer of a given univariate function with a Lipschitz gradient in an interval.

This version is for the case where all function/derivative information is available by function calls.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- userdata

		- is a structure that allows data to be passed into the function and derivative evaluation programs (see below).

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  On initial entry, status must be set to 1.
		  
		  Possible exit values are:
		  
		  * **0**
                    The run was successful
		  
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
		  
		  * **-7**
                    The objective function appears to be unbounded from
                    below
		  
		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-40**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.

	*
		- x

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the value of the approximate global minimizer $x$ after a successful (status = 0) call.

	*
		- f

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the the value of the objective function $f(x)$ at the approximate global minimizer $x$ after a successful (status = 0) call.

	*
		- g

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the the value of the gradient of the objective function $f^{\prime}(x)$ at the approximate global minimizer $x$ after a successful (status = 0) call.

	*
		- h

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the the value of the second derivative of the objective function $f^{\prime\prime}(x)$ at the approximate global minimizer $x$ after a successful (status = 0) call.

	*
		- eval_fgh

		- 
		  is a user-provided function that must have the following signature:
		  
		  .. ref-code-block:: cpp
		  
		  	:ref:`ipc_<doxid-galahad__ipc_8h_>` eval_fgh( :ref:`rpc_<doxid-galahad__rpc_8h_>` x,
		  	              :ref:`rpc_<doxid-galahad__rpc_8h_>` *f,
		  	              :ref:`rpc_<doxid-galahad__rpc_8h_>` *g,
		  	              :ref:`rpc_<doxid-galahad__rpc_8h_>` *h,
		  	              const void *userdata)
		  
		  The value of the objective function $f(x)$ and its first derivative $f^{\prime}(x)$ evaluated at x= $x$ must be assigned to f and g respectively, and the function return value set to 0. In addition, if control.second_derivatives_available has been set to true, when calling ugo_import, the user must also assign the value of the second derivative $f^{\prime\prime}(x)$ in h; it need not be assigned otherwise. If the evaluation is impossible at x, return should be set to a nonzero value.

.. index:: pair: function; ugo_solve_reverse
.. _doxid-galahad__ugo_8h_1a0b8f123f8e67bb0cb8a27c5ce87c824c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_solve_reverse(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *eval_status,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* x,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* f,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* g,
		:ref:`rpc_<doxid-galahad__rpc_8h_>`* h
	)

Find an approximation to the global minimizer of a given univariate function with a Lipschitz gradient in an interval.

This version is for the case where function/derivative information is only available by returning to the calling procedure.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the entry and exit status from the package.
		  
		  On initial entry, status must be set to 1.
		  
		  Possible exit values are:
		  
		  * **0**
                    The run was successful
		  
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
		  
		  * **-7**
                    The objective function appears to be unbounded from
                    below
		  
		  * **-18**
                    Too many iterations have been performed. This may
                    happen if control.maxit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-19**
                    The CPU time limit has been reached. This may happen
                    if control.cpu_time_limit is too small, but may also
                    be symptomatic of a badly scaled problem.
		  
		  * **-40**
                    The user has forced termination of solver by
                    removing the file named control.alive_file from unit
                    unit control.alive_unit.
		  
		  * **3**
                    The user should compute the objective function value
                    $f(x)$ and its first derivative
                    $f^{\prime}(x)$, and then re-enter the
                    function. The required values should be set in f and
                    g respectively, and eval_status (below) should be
                    set to 0. If the user is unable to evaluate
                    $f(x)$ or $f^{\prime}(x)$ - for
                    instance, if the function or its first derivative
                    are undefined at x - the user need not set f or g,
                    but should then set eval_status to a non-zero
                    value. This value can only occur when
                    control.second_derivatives_available = false.
		  
		  * **4**
                    The user should compute the objective function value
                    $f(x)$ and its first two derivatives
                    $f^{\prime}(x)$ and
                    $f^{\prime\prime}(x)$ at x= $x$, and
                    then re-enter the function. The required values
                    should be set in f, g and h respectively, and
                    eval_status (below) should be set to 0. If the user
                    is unable to evaluate $f(x)$,
                    $f^{\prime}(x)$ or
                    $f^{\prime\prime}(x)$ - for instance, if the
                    function or its derivatives are undefined at x - the
                    user need not set f, g or h, but should then set
                    eval_status to a non-zero value. This value can only
                    occur when control.second_derivatives_available =
                    true.

	*
		- eval_status

		- is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that is used to indicate if objective function and its derivatives can be provided (see above).

	*
		- x

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the next value of $x$ at which the user is required to evaluate the objective (and its derivatives) when status > 0, or the value of the approximate global minimizer when status = 0

	*
		- f

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that must be set by the user to hold the value of $f(x)$ if required by status > 0 (see above), and will return the value of the approximate global minimum when status = 0

	*
		- g

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that must be set by the user to hold the value of $f^{\prime}(x)$ if required by status > 0 (see above), and will return the value of the first derivative of $f$ at the approximate global minimizer when status = 0

	*
		- h

		- is a scalar variable of type :ref:`rpc_<doxid-galahad__rpc_8h_>`, that must be set by the user to hold the value of $f^{\prime\prime}(x)$ if required by status > 0 (see above), and will return the value of the second derivative of $f$ at the approximate global minimizer when status = 0

.. index:: pair: function; ugo_information
.. _doxid-galahad__ugo_8h_1a8e1db35daea3247b2cc9eb8607d0abee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_information(void **data, struct :ref:`ugo_inform_type<doxid-structugo__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a struct containing output information (see :ref:`ugo_inform_type <doxid-structugo__inform__type>`)

	*
		- status

		- 
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit status from the package. Possible values are (currently):
		  
		  * **0**
                    The values were recorded successfully

.. index:: pair: function; ugo_terminate
.. _doxid-galahad__ugo_8h_1ad9485926c547bb783aea3ee1adb3b084:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void ugo_terminate(
		void **data,
		struct :ref:`ugo_control_type<doxid-structugo__control__type>`* control,
		struct :ref:`ugo_inform_type<doxid-structugo__inform__type>`* inform
	)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a struct containing control information (see :ref:`ugo_control_type <doxid-structugo__control__type>`)

	*
		- inform

		- is a struct containing output information (see :ref:`ugo_inform_type <doxid-structugo__inform__type>`)

