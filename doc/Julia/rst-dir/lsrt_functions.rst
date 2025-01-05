callable functions
------------------

.. index:: pair: function; lsrt_initialize
.. _doxid-galahad__lsrt_8h_1a9c5c14ddb34a5ea1becd133837da6544:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; lsrt_read_specfile
.. _doxid-galahad__lsrt_8h_1a07c4c60e1ab6ae67a4da710e2ed01ff0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/lsrt/LSRT.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/lsrt.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; lsrt_import_control
.. _doxid-galahad__lsrt_8h_1a09e39db33990f0c8a66480f54ba80f09:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_import_control(T, INT, control, data, status)

Import control parameters prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; lsrt_solve_problem
.. _doxid-galahad__lsrt_8h_1aa1b3479d5f21fe373ef8948d55763992:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_solve_problem(T, INT, data, status, m, n, power, weight, x, u, v)

Solve the regularized least-squuares problem using reverse communication.

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

		  This must be set to

		  * **1**
                    on initial entry. Set the argument ``u`` (below) to
                    $b$ for this entry.

		  Possible exit values are:

		  * **0**
                    the solution has been found

		  * **2**
                    The user must perform the operation $$u := u + Av,$$
		    and recall the function. The vectors $u$ and
		    $v$ are available in the arrays ``u`` and ``v``
		    (below) respectively, and the result $u$ must overwrite
		    the content of ``u``. No argument except ``u`` should be
		    altered before recalling the function

		  * **3**
                    The user must perform the operation $$v := v + A^Tu,$$
		    and recall the function. The vectors $u$ and
		    $v$ are available in the arrays ``u`` and ``v``
		    respectively, and the result $v$ must overwrite the
		    content of ``v``. No argument except ``v`` should be
		    altered before recalling the function

		  * **4**
                    The user must reset ``u`` to $b$ are recall the
                    function. No argument except ``u`` should be altered
                    before recalling the function

		  * **-1**
                    an array allocation has failed

		  * **-2**
                    an array deallocation has failed

		  * **-3**
                    one or more of n, m, power or weight violates
                    allowed bounds

		  * **-18**
                    the iteration limit has been exceeded

		  * **-25**
                    status is negative on entry

	*
		- m

		- is a scalar variable of type INT that holds the number of equations (i.e., rows of $A$), $m > 0$

	*
		- n

		- is a scalar variable of type INT that holds the number of variables (i.e., columns of $A$), $n > 0$

	*
		- power

		- is a scalar of type T that holds the regularization power, $p \geq 2$

	*
		- weight

		- is a scalar of type T that holds the regularization weight, $\sigma > 0$

	*
		- x

		- is a one-dimensional array of size n and type T that holds the solution $x$. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- u

		- is a one-dimensional array of size m and type T that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type T that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; lsrt_information
.. _doxid-galahad__lsrt_8h_1ad3895aabdb7f18f84d209b02287872be:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lsrt_inform_type <doxid-structlsrt__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lsrt_terminate
.. _doxid-galahad__lsrt_8h_1ac3a3d73e2686538802563c795a1afff4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lsrt_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lsrt_control_type <doxid-structlsrt__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lsrt_inform_type <doxid-structlsrt__inform__type>`)
