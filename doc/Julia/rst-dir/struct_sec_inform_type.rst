.. index:: pair: table; sec_inform_type
.. _doxid-structsec__inform__type:

sec_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sec_inform_type{INT}
          status::INT

.. _details-structsec__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsec__inform__type_status:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. Possible values are:

* **0** 

  A successful return

* **-85**

  An update is inappropriate and has been skipped

