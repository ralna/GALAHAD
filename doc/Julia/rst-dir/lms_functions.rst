callable functions
------------------

.. index:: pair: function; lms_initialize
.. _doxid-galahad__lms_8h_1a9abec0f0f82474e01c99ce43ab9252f5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lms_initialize(T, INT, data, control, status)

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lms_control_type <doxid-structlms__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; lms_information
.. _doxid-galahad__lms_8h_1a0c692aa607e53b87fd8a1a8de116f5aa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lms_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`lms_inform_type <doxid-structlms__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; lms_terminate
.. _doxid-galahad__lms_8h_1a6c036818c80d8e54dcf4d0e7bb341e33:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function lms_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`lms_control_type <doxid-structlms__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`lms_inform_type <doxid-structlms__inform__type>`)
