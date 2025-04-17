.. index:: pair: table; bsc_inform_type
.. _doxid-structbsc__inform__type:

bsc_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct bsc_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          max_col_a::INT
          exceeds_max_col::INT
          time::T
          clock_time::T

.. _details-structbsc__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbsc__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

the return status from the package. Possible values are:

* **0**

  The call was succcesful

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

* **-3**

  The restrictions n > 0 or m > 0 or requirement that a type contains
  its relevant string 'dense', 'coordinate' or 'sparse_by_rows' has been
  violated.

.. index:: pair: variable; alloc_status
.. _doxid-structbsc__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbsc__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred.

.. index:: pair: variable; max_col_a
.. _doxid-structbsc__inform__type_max_col_a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_col_a

the maximum number of entries in a column of $A$

.. index:: pair: variable; exceeds_max_col
.. _doxid-structbsc__inform__type_exceeds_max_col:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT exceeds_max_col

the number of columns of $A$ that have more than control.max_col entries

.. index:: pair: variable; time
.. _doxid-structbsc__inform__type_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T time

the total CPU time spent in the package

.. index:: pair: variable; clock_time
.. _doxid-structbsc__inform__type_clock_time:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T clock_time

the total clock time spent in the package

