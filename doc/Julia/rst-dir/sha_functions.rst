callable functions
------------------

.. index:: pair: function; sha_initialize
.. _doxid-galahad__sha_8h_initialize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		-
		  is a structure containing control information (see
		  :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; sha_read_specfile
.. _doxid-galahad__sha_8h_read_specfile:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_read_specfile(control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/sha/SHA.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/sha.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		-
		  is a structure containing control information (see
		  :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- specfile

		-
		  is a one-dimensional array of type Vararg{Cchar} that
		  must give the name of the specification file



.. index:: pair: function; sha_analyse_matrix
.. _doxid-galahad__sha_8h_analyse_matrix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_analyse_matrix(T, INT, control, data, status, n, ne, row, col, m)

Analsyse the sparsity structure of $H$ to generate information that will be
used when estimating its values.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		-
		  is a structure whose members provide control paramters
		  for the remaining prcedures (see
		  :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package.

		  Possible values are:

		  * **0**
                    The import and analysis were conducted successfully.

		  * **1**
                    Insufficient data pairs $(s_i,y_i)$ have been provided,
                    as m is too small. The returned $B$ is likely not fully
                    accurate.

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
                    A restriction n > 0, ne $\geq$ 0 or 0 $\leq$ row[i] $\leq$
                    col[i] $\leq$ n has been violated.

	*
		- n

		-
		  is a scalar variable of type INT, that holds the
		  number of variables

	*
		- ne

		-
		  is a scalar variable of type INT, that holds the
		  number of entries in the upper triangular part of $H$.

	*
		- row

		-
		  is a one-dimensional array of size ne and type INT,
		  that holds the row indices of the upper triangular
		  part of $H$.

	*
		- col

		-
		  is a one-dimensional array of size ne and type INT,
		  that holds the column indices of the upper triangular
		  part of $H$.

	*
		- m

		-
		  is a scalar variable of type INT, that gives the
		  minimum number of $(s^{(k)},y^{(k)})$ pairs that will
		  be needed to recover a good Hessian approximation.

.. index:: pair: function; sha_recover_matrix
.. _doxid-galahad__sha_8h_recover_matrix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_recover_matrix(T, INT, data, status, ne, m, ls1, ls2, strans,
                                    ly1, ly2, ytrans, val, order)

Estimate the nonzero entries of the Hessian $H$ by component-wise secant
approximation.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package.

		  Possible values are:

		  * **0**
                    The recovery was successful.

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

		  * **-31**
                    sha.recover_matrix has been called before
                    sha.analyse_matrix.

	*
		- ne

		- is a scalar variable of type INT that holds the number of entries in the lower triangular part of the symmetric matrix $A$.

	*
		- val

		- is a one-dimensional array of size ne and type T that holds the values of the entries of the lower triangular part of the symmetric matrix $A$ in any of the supported storage schemes.


	*
		- ne

		-
                  is a scalar variable of type INT, that holds the number
                  of entries in the upper triangular part of $H$.

	*
		- m_available

		-
                  is a scalar variable of type INT, that holds the
                  number of differences provided. Ideally this will be
                  as large as m as reported by sha_analyse_matrix, but
                  better still there should be a further
                  control.extra_differences to allow for unlikely
                  singularities.

	*
		- ls1

		-
                  is a scalar variable of type INT, that holds the
                  leading (first) dimension of the array strans.

	*
		- ls2

		-
                  is a scalar variable of type INT, that holds the
                  trailing (second) dimension of the array strans.

	*
		- strans

		-
                  is a two-dimensional array of size [ls1][ls2] and type
                  T, that holds the values of the vectors $\{s^{(k) T}\}$.
                  Component [$k$][$i$] should hold $s_i^{(k)}$.

	*
		- ly1

		-
                  is a scalar variable of type INT, that holds the
                  leading (first) dimension of the array ytrans.

	*
		- ly2

		-
                  is a scalar variable of type INT, that holds the
                  trailing (second) dimension of the array ytrans.

	*
		- ytrans

		-
                  is a two-dimensional array of size [ly1][ly2] and type
                  T, that holds the values of the vectors $\{y^{(k) T}\}$.
                  Component [$k$][$i$] should hold $y_i^{(k)}$.

	*
		- val

		-
                  is a one-dimensional array of size ne and type T,
                  that holds the values of the entries of the upper
                  triangular part of the symmetric matrix $H$ in the
                  sparse coordinate scheme.

	*
		- order

		-
                  is a one-dimensional array of size m and type INT,
                  that holds the preferred order of access for the pairs
                  $\{(s^{(k)},y^{(k)})\}$. The $k$-th component of
                  order specifies the row number of strans and
                  ytrans that will be used as the $k$-th most
                  favoured. order need not be set if the natural
                  order, $k, k = 1,...,$ m, is desired, and this case
                  order should be C_NULL.

.. index:: pair: function; sha_information
.. _doxid-galahad__sha_8h_information:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		-
		  is a structure containing output information (see
		  :ref:`sha_inform_type <doxid-structsha__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sha_terminate
.. _doxid-galahad__sha_8h_terminate:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		-
		  is a structure containing control information (see
		  :ref:`sha_control_type  <doxid-structsha__control__type>`)

	*
		- inform

		-
		  is a structure containing output information (see
		  :ref:`sha_inform_type <doxid-structsha__inform__type>`)
