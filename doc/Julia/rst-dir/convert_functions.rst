callable functions
------------------

.. index:: pair: function; convert_initialize
.. _doxid-galahad__convert_8h_1a6b9f4c72cb9c23cae9d4900816685ad6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function convert_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`convert_control_type <doxid-structconvert__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; convert_information
.. _doxid-galahad__convert_8h_1aba73d8119f28983efa000812bd970be6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function convert_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`convert_inform_type <doxid-structconvert__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; convert_terminate
.. _doxid-galahad__convert_8h_1a98d7b586061edf44052ae76b80cd2697:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function convert_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`convert_control_type <doxid-structconvert__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`convert_inform_type <doxid-structconvert__inform__type>`)
