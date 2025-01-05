callable functions
------------------

.. index:: pair: function; roots_initialize
.. _doxid-galahad__roots_8h_1ac504c30e3b55ef425516ad7cf1638a76:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function roots_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`roots_control_type <doxid-structroots__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; roots_information
.. _doxid-galahad__roots_8h_1ac63ef195952ae821d6966a8f25ac2513:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function roots_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`roots_inform_type <doxid-structroots__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; roots_terminate
.. _doxid-galahad__roots_8h_1a4e81f4ac6c1119dfeb3a81729c3ec997:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function roots_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`roots_control_type <doxid-structroots__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`roots_inform_type <doxid-structroots__inform__type>`)
