.. index:: pair: table; convert_control_type
.. _doxid-structconvert__control__type:

convert_control_type structure
------------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct convert_control_type{INT}
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          transpose::Bool
          sum_duplicates::Bool
          order::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structconvert__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structconvert__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structconvert__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

unit for error messages

.. index:: pair: variable; out
.. _doxid-structconvert__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

unit for monitor output

.. index:: pair: variable; print_level
.. _doxid-structconvert__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

controls level of diagnostic output

.. index:: pair: variable; transpose
.. _doxid-structconvert__control__type_transpose:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool transpose

obtain the transpose of the input matrix?

.. index:: pair: variable; sum_duplicates
.. _doxid-structconvert__control__type_sum_duplicates:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool sum_duplicates

add the values of entries in duplicate positions?

.. index:: pair: variable; order
.. _doxid-structconvert__control__type_order:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool order

order row or column data by increasing index?

.. index:: pair: variable; space_critical
.. _doxid-structconvert__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if space is critical, ensure allocated arrays are no bigger than needed

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structconvert__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

exit if any deallocation fails

.. index:: pair: variable; prefix
.. _doxid-structconvert__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by prefix(2:LEN(TRIM(.prefix))-1) where prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

