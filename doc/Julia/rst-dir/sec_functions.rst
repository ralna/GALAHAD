callable functions
------------------

.. index:: pair: function; sec_initialize
.. _doxid-galahad__sec_8h_1adf7e7f81c32214d1e79170023d5d47e5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sec_initialize(T, INT, control, status)

Set default control values and initialize private data

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`sec_control_type <doxid-structsec__control__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The initialization was successful.

.. index:: pair: function; sec_information
.. _doxid-galahad__sec_8h_1a24da7562aed5f631b7c1e5442326f66e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sec_information(T, INT, data, inform, status)

Provides output information

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`sec_inform_type <doxid-structsec__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; sec_terminate
.. _doxid-galahad__sec_8h_1aff9c49fd2dea47f495445d0f357a8b19:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function sec_terminate(T, INT, data, control, inform)

Deallocate all internal private storage

.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`sec_control_type <doxid-structsec__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`sec_inform_type <doxid-structsec__inform__type>`)
