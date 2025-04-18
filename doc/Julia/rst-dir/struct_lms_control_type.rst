.. index:: pair: table; lms_control_type
.. _doxid-structlms__control__type:

lms_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lms_control_type{INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          memory_length::INT
          method::INT
          any_method::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structlms__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structlms__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structlms__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structlms__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structlms__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; memory_length
.. _doxid-structlms__control__type_memory_length:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT memory_length

limited memory length

.. index:: pair: variable; method
.. _doxid-structlms__control__type_method:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT method

limited-memory formula required (others may be added in due course):

* 1 BFGS (default).

* 2 Symmetric Rank-One (SR1).

* 3 The inverse of the BFGS formula.

* 4 The inverse of the shifted BFGS formula. This should be used instead of .method = 3 whenever a shift is planned.

.. index:: pair: variable; any_method
.. _doxid-structlms__control__type_any_method:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool any_method

allow space to permit different methods if required (less efficient)

.. index:: pair: variable; space_critical
.. _doxid-structlms__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structlms__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structlms__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

