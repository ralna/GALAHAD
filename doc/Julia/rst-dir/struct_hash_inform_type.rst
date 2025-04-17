.. index:: pair: table; hash_inform_type
.. _doxid-structhash__inform__type:

hash_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct hash_inform_type{INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}

.. _details-structhash__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structhash__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. Possible values are:

* **0**

  The initialization, insertion or deletion was successful.

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

* **-99**

  The current dictionary is full and should be rebuilt with more space.

.. index:: pair: variable; alloc_status
.. _doxid-structhash__inform__type_alloc_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation.

.. index:: pair: variable; bad_alloc
.. _doxid-structhash__inform__type_bad_alloc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred.

