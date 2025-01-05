callable functions
------------------

.. index:: pair: function; bsc_initialize
.. _doxid-galahad__bsc_8h_1a32dd948f5ce268b0cdb340f435819c8e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_initialize(T, INT, data, control, status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; bsc_information
.. _doxid-galahad__bsc_8h_1a4c0ae2150d39c240539e1d3be836b0af:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; bsc_terminate
.. _doxid-galahad__bsc_8h_1a3a8a2f875e681225b4851d060e310271:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function bsc_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`bsc_control_type <doxid-structbsc__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`bsc_inform_type <doxid-structbsc__inform__type>`)
