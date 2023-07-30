.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_scu_control_type.rst
	struct_scu_inform_type.rst

function calls
--------------

.. index:: pair: function; scu_information
.. _doxid-galahad__scu_8h_1ad52752848139c1772e7d5bb4aa2a3f6d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void scu_information(void** data, structure :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform, int* status)

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

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; scu_terminate
.. _doxid-galahad__scu_8h_1a6fc2d5a0cb41e7c912661c5101d2ffad:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void scu_terminate(
		void** data,
		struct :ref:`scu_control_type<doxid-structscu__control__type>`* control,
		struct :ref:`scu_inform_type<doxid-structscu__inform__type>`* inform
	)

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
