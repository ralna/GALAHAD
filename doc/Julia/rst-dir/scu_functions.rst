callable functions
------------------

.. index:: pair: function; scu_information
.. _doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function scu_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`scu_inform_type <doxid-structscu__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; scu_terminate
.. _doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function scu_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`scu_control_type <doxid-structscu__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`scu_inform_type <doxid-structscu__inform__type>`)
