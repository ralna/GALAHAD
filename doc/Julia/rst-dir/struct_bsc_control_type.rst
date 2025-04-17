.. index:: pair: table; bsc_control_type
.. _doxid-structbsc__control__type:

bsc_control_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bsc_control_type
          f_indexing::Bool
          error::INT
          out::INT
          print_level::INT
          max_col::INT
          new_a::INT
          extra_space_s::INT
          s_also_by_column::Bool
          space_critical::Bool
          deallocate_error_fatal::Bool
          prefix::NTuple{31,Cchar}

.. _details-structbsc__control__type:

detailed documentation
----------------------

control derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; f_indexing
.. _doxid-structbsc__control__type_f_indexing:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool f_indexing

use C or Fortran sparse matrix indexing

.. index:: pair: variable; error
.. _doxid-structbsc__control__type_error:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT error

error and warning diagnostics occur on stream error

.. index:: pair: variable; out
.. _doxid-structbsc__control__type_out:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out

general output occurs on stream out

.. index:: pair: variable; print_level
.. _doxid-structbsc__control__type_print_level:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT print_level

the level of output required is specified by print_level

.. index:: pair: variable; max_col
.. _doxid-structbsc__control__type_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col

maximum permitted number of nonzeros in a column of $A$; -ve means unlimit

.. index:: pair: variable; new_a
.. _doxid-structbsc__control__type_new_a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT new_a

how much has $A$ changed since it was last accessed:

* 0 = not changed,

* 1 = values changed,

* 2 = structure changed

* 3 = structure changed but values not required

.. index:: pair: variable; extra_space_s
.. _doxid-structbsc__control__type_extra_space_s:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT extra_space_s

how much extra space is to be allocated in $S$ above that needed to hold the Schur complement

.. index:: pair: variable; s_also_by_column
.. _doxid-structbsc__control__type_s_also_by_column:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool s_also_by_column

should s.ptr also be set to indicate the first entry in each column of $S$

.. index:: pair: variable; space_critical
.. _doxid-structbsc__control__type_space_critical:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool space_critical

if .space_critical true, every effort will be made to use as little space as possible. This may result in longer computation time

.. index:: pair: variable; deallocate_error_fatal
.. _doxid-structbsc__control__type_deallocate_error_fatal:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool deallocate_error_fatal

if .deallocate_error_fatal is true, any array/pointer deallocation error will terminate execution. Otherwise, computation will continue

.. index:: pair: variable; prefix
.. _doxid-structbsc__control__type_prefix:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{31,Cchar} prefix

all output lines will be prefixed by .prefix(2:LEN(TRIM(.prefix))-1) where .prefix contains the required string enclosed in quotes, e.g. "string" or 'string'

