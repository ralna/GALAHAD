callable functions
------------------

.. index:: pair: function; nodend_initialize
.. _doxid-galahad__nodend_initialize:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nodend_initialize(INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`nodend_control_type <doxid-structnodend__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; nodend_information
.. _doxid-galahad__nodend_information:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function nodend_information(INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`nodend_inform_type <doxid-structnodend__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

