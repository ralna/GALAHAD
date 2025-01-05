callable functions
------------------

.. index:: pair: function; ir_initialize
.. _doxid-galahad__ir_8h_1a1da2baeef0fe4c8e8937674a0c491c14:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function ir_initialize(T, INT, data, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`ir_control_type <doxid-structir__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; ir_information
.. _doxid-galahad__ir_8h_1a744223166561eeb7bf4f9e6d65d2e991:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function ir_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`ir_inform_type <doxid-structir__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; ir_terminate
.. _doxid-galahad__ir_8h_1aa7a05607f24ba9e539d96c889aca1134:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function ir_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`ir_control_type <doxid-structir__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`ir_inform_type <doxid-structir__inform__type>`)
