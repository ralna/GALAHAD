callable functions
------------------

.. index:: pair: function; hash_initialize
.. _doxid-galahad__hash_8h_1ac983b0236ce2f2ae9ed016846c5ad2a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function hash_initialize(T, INT, nchar, length, data, control, inform)

Set default control values and initialize private data



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- nchar

		- is a scalar variable of type INT that holds the number of characters permitted in each word in the hash table

	*
		- length

		- is a scalar variable of type INT that holds the maximum number of words that can be held in the dictionary

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`hash_control_type <doxid-structhash__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`hash_inform_type <doxid-structhash__inform__type>`)

.. index:: pair: function; hash_information
.. _doxid-galahad__hash_8h_1a7f73a5ca2bbdc3af1b7793f7b14ed13f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function hash_information(T, INT, data, inform, status)

Provides output information



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- inform

		- is a structure containing output information (see :ref:`hash_inform_type <doxid-structhash__inform__type>`)

	*
		- status

		- is a scalar variable of type INT that gives the exit
		  status from the package. Possible values are
		  (currently):

		  * **0**
                    The values were recorded successfully

.. index:: pair: function; hash_terminate
.. _doxid-galahad__hash_8h_1a0aece137337307f3c98e9b201205170d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

        function hash_terminate(T, INT, data, control, inform)

Deallocate all internal private storage



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data

		- holds private internal data

	*
		- control

		- is a structure containing control information (see :ref:`hash_control_type <doxid-structhash__control__type>`)

	*
		- inform

		- is a structure containing output information (see :ref:`hash_inform_type <doxid-structhash__inform__type>`)
