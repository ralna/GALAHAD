.. _global:

overview of functions provided
------------------------------

.. toctree::
	:hidden:

	struct_sec_control_type.rst
	struct_sec_inform_type.rst

function calls
--------------

.. index:: pair: function; sec_initialize
.. _doxid-galahad__sec_8h_1adf7e7f81c32214d1e79170023d5d47e5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void sec_initialize(struct :ref:`sec_control_type<doxid-structsec__control__type>`* control, int* status)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- control

		- is a structure containing control information (see :ref:`sec_control_type <doxid-structsec__control__type>`)

	*
		- status

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The initialization was succesful.

.. index:: pair: function; sec_information
.. _doxid-galahad__sec_8h_1a24da7562aed5f631b7c1e5442326f66e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void sec_information(void** data, structure :ref:`sec_inform_type<doxid-structsec__inform__type>`* inform, int* status)

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

		-
		  is a scalar variable of type Int32 that gives the exit status from the package. Possible values are (currently):

		  * 0. The values were recorded succesfully

.. index:: pair: function; sec_terminate
.. _doxid-galahad__sec_8h_1aff9c49fd2dea47f495445d0f357a8b19:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	void sec_terminate(
		void** data,
		struct :ref:`sec_control_type<doxid-structsec__control__type>`* control,
		struct :ref:`sec_inform_type<doxid-structsec__inform__type>`* inform
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

		- is a structure containing control information (see :ref:`sec_control_type <doxid-structsec__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`sec_inform_type <doxid-structsec__inform__type>`)
