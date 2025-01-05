callable functions
------------------

.. index:: pair: function; lstr_initialize
.. _doxid-galahad__lstr_8h_1ae423bf7ffc77c89f461448ca1f5c286c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lstr_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; lstr_read_specfile
.. _doxid-galahad__lstr_8h_1a3d3fa989fe4c3b40cd7e296249d2205d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lstr_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/lstr/LSTR.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/lstr.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; lstr_import_control
.. _doxid-galahad__lstr_8h_1a1a8ad63d944dc046fd2040554d6d01e5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lstr_import_control(T, INT, control, data, status)

Import control parameters prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

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

.. index:: pair: function; lstr_solve_problem
.. _doxid-galahad__lstr_8h_1af3355e5a8df63a9c7173eb974a1e7562:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lstr_solve_problem(T, INT, data, status, m, n, radius, x, u, v)

Solve the trust-region least-squares problem using reverse communication.

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
                    on initial entry. Set the argument ``u`` (below) to $b$
                    for this entry.

		  * **5**
                    the iteration is to be restarted with a smaller
                    radius but with all other data unchanged. Set ``u``
                    to $b$ for this entry.

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
                    one or more of n, m or weight violates allowed
                    bounds

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
		- radius

		- is a scalar of type T that holds the trust-region radius, $\Delta > 0$

	*
		- x

		- is a one-dimensional array of size n and type T that holds the solution $x$. The j-th component of ``x``, j = 1, ... , n, contains $x_j$.

	*
		- u

		- is a one-dimensional array of size m and type T that should be used and reset appropriately when status = 1 to 5 as directed by status.

	*
		- v

		- is a one-dimensional array of size n and type T that should be used and reset appropriately when status = 1 to 5 as directed by status.

.. index:: pair: function; lstr_information
.. _doxid-galahad__lstr_8h_1a5929f00ea00af253ede33a6749451481:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lstr_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lstr_inform_type <doxid-structlstr__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lstr_terminate
.. _doxid-galahad__lstr_8h_1aa198189942e179e52699e1fedfcdf9d1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lstr_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lstr_control_type <doxid-structlstr__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lstr_inform_type <doxid-structlstr__inform__type>`)
