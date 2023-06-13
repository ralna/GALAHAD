.. index:: pair: table; cro_inform_type
.. _doxid-structcro__inform__type:

cro_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	cro_inform_type = {
		-- fields
	
		:ref:`status<doxid-structcro__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structcro__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`bad_alloc<doxid-structcro__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`dependent<doxid-structcro__inform__type_1a3678dbffc0e2f3521f7ef27194b21ab6>`,
		:ref:`time<doxid-structcro__inform__type_1a0d99b2a30c1bf487fddf2643b03a3120>`,
		:ref:`sls_inform<doxid-structcro__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`,
		:ref:`sbls_inform<doxid-structcro__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`,
		:ref:`uls_inform<doxid-structcro__inform__type_1aa39eb0d7b50d4a858849f8ef652ae84c>`,
		:ref:`scu_status<doxid-structcro__inform__type_1a25bf1e7f86c2b4f4836aa4de40019815>`,
		:ref:`scu_inform<doxid-structcro__inform__type_1a0b702af94f05b9d4bb2bb6416f2498ee>`,
		:ref:`ir_inform<doxid-structcro__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5>`,
	}

.. _details-structcro__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structcro__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

return status. See CRO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structcro__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structcro__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; dependent
.. _doxid-structcro__inform__type_1a3678dbffc0e2f3521f7ef27194b21ab6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	dependent

the number of dependent active constraints

.. index:: pair: variable; time
.. _doxid-structcro__inform__type_1a0d99b2a30c1bf487fddf2643b03a3120:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time

timings (see above)

.. index:: pair: variable; sls_inform
.. _doxid-structcro__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sls_inform

information from SLS

.. index:: pair: variable; sbls_inform
.. _doxid-structcro__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sbls_inform

information from SBLS

.. index:: pair: variable; uls_inform
.. _doxid-structcro__inform__type_1aa39eb0d7b50d4a858849f8ef652ae84c:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	uls_inform

information from ULS

.. index:: pair: variable; scu_status
.. _doxid-structcro__inform__type_1a25bf1e7f86c2b4f4836aa4de40019815:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	scu_status

information from SCU

.. index:: pair: variable; scu_inform
.. _doxid-structcro__inform__type_1a0b702af94f05b9d4bb2bb6416f2498ee:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	scu_inform

see scu_status

.. index:: pair: variable; ir_inform
.. _doxid-structcro__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	ir_inform

information from IR

