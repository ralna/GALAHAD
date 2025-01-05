callable functions
------------------

.. index:: pair: function; sils_initialize
.. _doxid-galahad__sils_8h_1adfa46fc519194d9acfbeccac4c5a1af3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sils_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sils_read_specfile
.. _doxid-galahad__sils_8h_1a12447d25d91610c87b4c8ce7744aefd7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sils_read_specfile(T, INT, control, specfile)

Read the content of a specification file, and assign values associated
with given keywords to the corresponding control parameters.  An
in-depth discussion of specification files is
:ref:`available<details-spec_file>`, and a detailed list of keywords
with associated default values is provided in
\$GALAHAD/src/sils/SILS.template.  See also Table 2.1 in the Fortran
documentation provided in \$GALAHAD/doc/sils.pdf for a list of how these
keywords relate to the components of the control structure.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- specfile

		- is a one-dimensional array of type Vararg{Cchar} that must give the name of the specification file

.. index:: pair: function; sils_import
.. _doxid-galahad__sils_8h_1a78d5647031a8a4522541064853b021ba:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sils_import(T, INT, control, data, status)

Import problem data into internal storage prior to solution.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`sils_control_type <doxid-structsils__control__type>`)

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

.. index:: pair: function; sils_reset_control
.. _doxid-galahad__sils_8h_1a34e5304b29c89525543cd512f426ac4f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sils_reset_control(T, INT, control, data, status)

Reset control parameters after import if required.

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure whose members provide control parameters for the remaining procedures (see :ref:`sils_control_type <doxid-structsils__control__type>`)

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

.. index:: pair: function; sils_information
.. _doxid-galahad__sils_8h_1a27320b6d18c7508283cfb19dc8fecf37:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sils_information(T, INT, data, ainfo, finfo, sinfo, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- ainfo

		- is a structure containing output information (see :ref:`sils_ainfo_type <doxid-structsils__ainfo__type>`)

	*
		- finfo

		- is a structure containing output information (see :ref:`sils_finfo_type <doxid-structsils__finfo__type>`)

	*
		- sinfo

		- is a structure containing output information (see :ref:`sils_sinfo_type <doxid-structsils__sinfo__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sils_finalize
.. _doxid-galahad__sils_8h_1aa862612cd37fce35b1d35bd6ad295d82:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sils_finalize(T, INT, data, control, status)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sils_control_type <doxid-structsils__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

		  * $\neq$ **0**
                    The Fortran STAT value of a deallocate statement that has failed
