.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_gls_control.rst
	struct_gls_ainfo.rst
	struct_gls_finfo.rst
	struct_gls_sinfo.rst

function calls
--------------

.. index:: pair: function; gls_initialize
.. _doxid-galahad__gls_8h_1ab7827883517db347ee1229eda004ede5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void gls_initialize(void** data, structure :ref:`gls_control<doxid-structgls__control>`* control)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`gls_control <doxid-structgls__control>`)

.. index:: pair: function; gls_read_specfile
.. _doxid-galahad__gls_8h_1a428c3dcc1d0de87f6108d396eec6e176:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void gls_read_specfile(struct :ref:`gls_control<doxid-structgls__control>`* control, const Vararg{Cchar} specfile[])

Read the content of a specification file, and assign values associated with given keywords to the corresponding control parameters. By default, the spcification file will be named RUNGLS.SPC and lie in the current directory. Refer to Table 2.1 in the fortran documentation provided in $GALAHAD/doc/gls.pdf for a list of keywords that may be set.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`gls_control <doxid-structgls__control>`)

	*
		- specfile

		- is a character string containing the name of the specification file

.. index:: pair: function; gls_import
.. _doxid-galahad__gls_8h_1a1b34338e803f603af4161082a25f4e58:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void gls_import(struct :ref:`gls_control<doxid-structgls__control>`* control, void** data, int* status)

Import problem data into internal storage prior to solution.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`gls_control <doxid-structgls__control>`)

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

.. index:: pair: function; gls_reset_control
.. _doxid-galahad__gls_8h_1a8b84f081ccc0b05b733adc2f0a829c07:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void gls_reset_control(struct :ref:`gls_control<doxid-structgls__control>`* control, void** data, int* status)

Reset control parameters after import if required.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control paramters for the remaining prcedures (see :ref:`gls_control <doxid-structgls__control>`)

	*
		- data

		- holds private internal data

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are:

		  * 1. The import was succesful, and the package is ready for the solve phase

.. index:: pair: function; gls_information
.. _doxid-galahad__gls_8h_1a620dc0f7a9ef6049a7bafdc02913da47:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void gls_information(
		void** data,
		struct :ref:`gls_ainfo<doxid-structgls__ainfo>`* ainfo,
		struct :ref:`gls_finfo<doxid-structgls__finfo>`* finfo,
		struct :ref:`gls_sinfo<doxid-structgls__sinfo>`* sinfo,
		int* status
	)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- ainfo

		- is a structure containing analysis output information (see :ref:`gls_ainfo <doxid-structgls__ainfo>`)

	*
		- finfo

		- is a structure containing factorization output information (see :ref:`gls_finfo <doxid-structgls__finfo>`)

	*
		- sinfo

		- is a structure containing solver output information (see :ref:`gls_sinfo <doxid-structgls__sinfo>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; gls_finalize
.. _doxid-galahad__gls_8h_1a4758f1fc9cad110a33c1778254b51390:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void gls_finalize(void** data, structure :ref:`gls_control<doxid-structgls__control>`* control, int* status)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`gls_control <doxid-structgls__control>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully
