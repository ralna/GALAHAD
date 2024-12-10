..
  .. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_sha_control_type.rst
	struct_sha_inform_type.rst

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block


	// typedefs

	typedef float :ref:`spc_<doxid-galahad__spc_8h_>`;
	typedef double :ref:`rpc_<doxid-galahad__rpc_8h_>`;
	typedef int :ref:`ipc_<doxid-galahad__ipc_8h_>`;

	// structs

	struct :ref:`sha_control_type<doxid-structsha__control__type>`;
	struct :ref:`sha_inform_type<doxid-structsha__inform__type>`;

	// function calls

	void :ref:`sha_initialize<doxid-galahad__sha_8h_initialize>`(void **data, struct :ref:`sha_control_type<doxid-structsha__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`sha_read_specfile<doxid-galahad__sha_8h_read_specfile>`(struct :ref:`sha_control_type<doxid-structsha__control__type>`* control, const char specfile[]);

	void :ref:`sha_analyse_matrix<doxid-galahad__sha_8h_analyse_matrix>`(
		struct :ref:`sha_control_type<doxid-structsha__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` col[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *m
	);

	void :ref:`sha_recover_matrix<doxid-galahad__sha_8h_recover_matrix>`(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` m,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ls1,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ls2,
                const real_wp_ strans[][ls2],
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ly1,
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ly2,
                const real_wp_ ytrans[][ly2],
                real_wp_ val[],
                const :ref:`ipc_<doxid-galahad__ipc_8h_>` order[]
	);


	void :ref:`sha_information<doxid-galahad__sha_8h_information>`(void **data, struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status);

	void :ref:`sha_terminate<doxid-galahad__sha_8h_terminate>`(
		void **data,
		struct :ref:`sha_control_type<doxid-structsha__control__type>`* control,
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform
	);

..
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

.. index:: pair: function; sha_initialize
.. _doxid-galahad__sha_8h_initialize:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_initialize(void **data, struct :ref:`sha_control_type<doxid-structsha__control__type>`* control, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a struct containing control information (see
		  :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- status

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.


.. index:: pair: function; sha_read_specfile
.. _doxid-galahad__sha_8h_read_specfile:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_read_specfile(struct :ref:`sha_control_type<doxid-structsha__control__type>`* control, const char specfile[])

Read the content of a specification file, and assign values
associated with given keywords to the corresponding control
parameters. An in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list
of keywords with associated default values is provided in
\$GALAHAD/src/xxx/XXX.template.
See also Table 2.1 in the Fortran documentation provided in
\$GALAHAD/doc/xxx.pdf for a list of how these keywords
relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		-
		  is a struct containing control information (see
		  :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- specfile

		-
		  is a character string containing the name of the
		  specification file

.. index:: pair: function; sha_analyse_matrix
.. _doxid-galahad__sha_8h_analyse_matrix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_analyse_matrix(
		struct :ref:`sha_control_type<doxid-structsha__control__type>`* control,
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` n,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` row[],
		const :ref:`ipc_<doxid-galahad__ipc_8h_>` col[],
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *m
	)

Analsyse the sparsity structure of $H$ to generate information that will be
used when estimating its values.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		-
		  is a struct whose members provide control paramters
		  for the remaining prcedures (see
		  :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit
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
                    A restriction n > 0, ne $\geq$ 0 or 0 $\leq$ row[i] $\leq$
                    col[i] $\leq$ n has been violated.

	*
		- n

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
		  number of variables

	*
		- ne

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
		  number of entries in the upper triangular part of $H$.

	*
		- row

		-
		  is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`,
		  that holds the row indices of the upper triangular
		  part of $H$.

	*
		- col

		-
		  is a one-dimensional array of size ne and type :ref:`ipc_<doxid-galahad__ipc_8h_>`,
		  that holds the column indices of the upper triangular
		  part of $H$.

	*
		- m

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the
		  minimum number of $(s^{(k)},y^{(k)})$ pairs that will
		  be needed to recover a good Hessian approximation.

.. index:: pair: function; sha_recover_matrix
.. _doxid-galahad__sha_8h_recover_matrix:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_recover_matrix(
		void **data,
		:ref:`ipc_<doxid-galahad__ipc_8h_>` *status,
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ne,
                :ref:`ipc_<doxid-galahad__ipc_8h_>` m_available,
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ls1,
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ls2,
                const real_wp_ strans[][ls2],
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ly1,
                :ref:`ipc_<doxid-galahad__ipc_8h_>` ly2,
                const real_wp_ ytrans[][ly2],
                real_wp_ val[],
                const :ref:`ipc_<doxid-galahad__ipc_8h_>` order[]
	)


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

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit
		  status from the package. Possible values are:

		  * **0**
                    The recovery was successful

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

		  * **-31**
                    sha_recover_matrix has been called before
                    sha_analyse_matrix.

	*
		- ne

		-
                  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the number
                  of entries in the upper triangular part of $H$.

	*
		- m_available

		-
                  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
                  number of differences provided. Ideally this will be
                  as large as m as reported by sha_analyse_matrix, but
                  better still there should be a further
                  control.extra_differences to allow for unlikely
                  singularities.

	*
		- ls1

		-
                  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
                  leading (first) dimension of the array strans.

	*
		- ls2

		-
                  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
                  trailing (second) dimension of the array strans.

	*
		- strans

		-
                  is a two-dimensional array of size [ls1][ls2] and type
                  :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the vectors $\{s^{(k) T}\}$.
                  Component [$k$][$i$] should hold $s_i^{(k)}$.

	*
		- ly1

		-
                  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
                  leading (first) dimension of the array ytrans.

	*
		- ly2

		-
                  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that holds the
                  trailing (second) dimension of the array ytrans.

	*
		- ytrans

		-
                  is a two-dimensional array of size [ly1][ly2] and type
                  :ref:`rpc_<doxid-galahad__rpc_8h_>`, that holds the values of the vectors $\{y^{(k) T}\}$.
                  Component [$k$][$i$] should hold $y_i^{(k)}$.

	*
		- val

		-
                  is a one-dimensional array of size ne and type :ref:`rpc_<doxid-galahad__rpc_8h_>`,
                  that holds the values of the entries of the upper
                  triangular part of the symmetric matrix $H$ in the
                  sparse coordinate scheme.

	*
		- order

		-
                  is a one-dimensional array of size m and type :ref:`ipc_<doxid-galahad__ipc_8h_>`,
                  that holds the preferred order of access for the pairs
                  $\{(s^{(k)},y^{(k)})\}$. The $k$-th component of
                  order specifies the row number of strans and
                  ytrans that will be used as the $k$-th most
                  favoured. order need not be set if the natural
                  order, $k, k = 1,...,$ m, is desired, and this case
                  order should be NULL.

.. index:: pair: function; sha_information
.. _doxid-galahad__sha_8h_information:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_information(void **data, struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform, :ref:`ipc_<doxid-galahad__ipc_8h_>` *status)

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
		  is a struct containing output information (see
		  :ref:`sha_inform_type <doxid-structsha__inform__type>`)

	*
		- status

		-
		  is a scalar variable of type :ref:`ipc_<doxid-galahad__ipc_8h_>`, that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sha_terminate
.. _doxid-galahad__sha_8h_terminate:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void sha_terminate(
		void **data,
		struct :ref:`sha_control_type<doxid-structsha__control__type>`* control,
		struct :ref:`sha_inform_type<doxid-structsha__inform__type>`* inform
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

		-
		  is a struct containing control information (see
		  :ref:`sha_control_type  <doxid-structsha__control__type>`)

	*
		- inform

		-
		  is a struct containing output information (see
		  :ref:`sha_inform_type <doxid-structsha__inform__type>`)
