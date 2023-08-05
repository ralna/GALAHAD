.. _global:

callable functions
------------------

.. index:: pair: function; sha_initialize
.. _doxid-galahad__sha_8h_1aa4f01a598c5cef45420937d4951519a9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_initialize(data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- status

		- is a scalar variable of type Int32 that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; sha_information
.. _doxid-galahad__sha_8h_1ace357a88ad000654a0ec0817d6d28ece:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_information(data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`sha_inform_type <doxid-structsha__inform__type>`)

	*
		- status

		- is a scalar variable of type Int32 that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sha_terminate
.. _doxid-galahad__sha_8h_1a9ea67bcd115e6479e7d93faf3445405a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sha_terminate(data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sha_control_type <doxid-structsha__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`sha_inform_type <doxid-structsha__inform__type>`)
