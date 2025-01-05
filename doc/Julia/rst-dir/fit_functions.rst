callable functions
------------------

.. index:: pair: function; fit_initialize
.. _doxid-galahad__fit_8h_1a53019c0890b67dbc8c8efa541e652a73:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fit_initialize(T, INT, data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`fit_control_type <doxid-structfit__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; fit_information
.. _doxid-galahad__fit_8h_1addba9b8a0adf58039c6974a4d30af840:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fit_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`fit_inform_type <doxid-structfit__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; fit_terminate
.. _doxid-galahad__fit_8h_1a92659f3983205f2d9694e555b2439390:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function fit_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`fit_control_type <doxid-structfit__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`fit_inform_type <doxid-structfit__inform__type>`)
