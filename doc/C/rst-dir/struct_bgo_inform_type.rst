.. index:: pair: table; bgo_inform_type
.. _doxid-structbgo__inform__type:

bgo_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	bgo_inform_type = {
		-- fields
	
		:ref:`status<doxid-structbgo__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structbgo__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`bad_alloc<doxid-structbgo__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`f_eval<doxid-structbgo__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa>`,
		:ref:`g_eval<doxid-structbgo__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469>`,
		:ref:`h_eval<doxid-structbgo__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a>`,
		:ref:`obj<doxid-structbgo__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`,
		:ref:`norm_pg<doxid-structbgo__inform__type_1acb02a4d1ae275a55874bb9897262b1fe>`,
		:ref:`time<doxid-structbgo__inform__type_1a323c159d2e08628b6db82791b80a2f30>`,
		:ref:`ugo_inform<doxid-structbgo__inform__type_1a51109e95a1bf5edbca5d7d1279b5a554>`,
		:ref:`lhs_inform<doxid-structbgo__inform__type_1aa1305fc54f2639b2f4c39c629b39cd48>`,
		:ref:`trb_inform<doxid-structbgo__inform__type_1a60ab8c5ff5dacc22bcaa60f4d6e8b321>`,
	}

.. _details-structbgo__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structbgo__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

return status. See BGO_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structbgo__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structbgo__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; f_eval
.. _doxid-structbgo__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structbgo__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structbgo__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; obj
.. _doxid-structbgo__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	obj

the value of the objective function at the best estimate of the solution determined by BGO_solve

.. index:: pair: variable; norm_pg
.. _doxid-structbgo__inform__type_1acb02a4d1ae275a55874bb9897262b1fe:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	norm_pg

the norm of the projected gradient of the objective function at the best estimate of the solution determined by BGO_solve

.. index:: pair: variable; time
.. _doxid-structbgo__inform__type_1a323c159d2e08628b6db82791b80a2f30:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time

timings (see above)

.. index:: pair: variable; ugo_inform
.. _doxid-structbgo__inform__type_1a51109e95a1bf5edbca5d7d1279b5a554:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	ugo_inform

inform parameters for UGO

.. index:: pair: variable; lhs_inform
.. _doxid-structbgo__inform__type_1aa1305fc54f2639b2f4c39c629b39cd48:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	lhs_inform

inform parameters for LHS

.. index:: pair: variable; trb_inform
.. _doxid-structbgo__inform__type_1a60ab8c5ff5dacc22bcaa60f4d6e8b321:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	trb_inform

inform parameters for TRB

