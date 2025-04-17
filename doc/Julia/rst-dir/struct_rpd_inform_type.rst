.. index:: pair: table; rpd_inform_type
.. _doxid-structrpd__inform__type:

rpd_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct rpd_inform_type{INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          io_status::INT
          line::INT
          p_type::NTuple{4,Cchar}

.. _details-structrpd__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structrpd__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. Possible values are:

* **0**

  The call was successful.

* **-1**

  An allocation error occurred. A message indicating the offending array
  is written on unit control.error, and the returned allocation status
  and a string containing the name of the offending array are held in
  inform.alloc_status and inform.bad_alloc respectively.

* **-2**

  A deallocation error occurred. A message indicating the offending
  array is written on unit control.error and the returned allocation
  status and a string containing the name of the offending array are
  held in inform.alloc_status and inform.bad_alloc respectively.

* **-22**

  An input/outpur error occurred.

* **-25**

  The end of the input file was reached prematurely.

* **-29**

  The problem type was not recognised.

.. index:: pair: variable; alloc_status
.. _doxid-structrpd__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation or deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structrpd__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation or deallocation error occurred

.. index:: pair: variable; io_status
.. _doxid-structrpd__inform__type_io_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT io_status

status from last read attempt

.. index:: pair: variable; line
.. _doxid-structrpd__inform__type_line:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT line

number of last line read from i/o file

.. index:: pair: variable; p_type
.. _doxid-structrpd__inform__type_p_type:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char p_type[4]

problem type

