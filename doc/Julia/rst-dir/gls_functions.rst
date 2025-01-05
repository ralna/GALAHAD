
callable functions
------------------

.. index:: pair: function; gls_initialize
.. _doxid-galahad__gls_8h_1ab7827883517db347ee1229eda004ede5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gls_initialize(T, INT, data, control)

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

        function gls_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/gls/GLS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/gls.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`gls_control <doxid-structgls__control>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; gls_import
.. _doxid-galahad__gls_8h_1a1b34338e803f603af4161082a25f4e58:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gls_import(T, INT, control, data, status)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`gls_control <doxid-structgls__control>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

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

		  * **-3**
                    The restriction n > 0 or requirement that type
                    contains its relevant string 'dense', 'coordinate',
                    'sparse_by_rows', 'diagonal' or 'absent' has been
                    violated.

.. index:: pair: function; gls_reset_control
.. _doxid-galahad__gls_8h_1a8b84f081ccc0b05b733adc2f0a829c07:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gls_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`gls_control <doxid-structgls__control>`)

	*
		- data

		- holds private internal data

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are:

		  * **1**
                    The import was successful, and the package is ready
                    for the solve phase

.. index:: pair: function; gls_information
.. _doxid-galahad__gls_8h_1a620dc0f7a9ef6049a7bafdc02913da47:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gls_information(T, INT, data, ainfo, finfo, sinfo, status)

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

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; gls_finalize
.. _doxid-galahad__gls_8h_1a4758f1fc9cad110a33c1778254b51390:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function gls_finalize(T, INT, data, control, status)

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

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully
