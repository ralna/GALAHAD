.. index:: pair: table; lstr_inform_type
.. _doxid-structlstr__inform__type:

lstr_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. _details-structlstr__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: table; lstr_inform_type
.. _doxid-structlstr__inform__type:

table lstr_inform_type
======================


.. toctree::
	:hidden:

Overview
~~~~~~~~

inform derived type as a C struct :ref:`More...<details-structlstr__inform__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	lstr_inform_type = {
		-- fields
	
		:ref:`status<doxid-structlstr__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structlstr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`bad_alloc<doxid-structlstr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`iter<doxid-structlstr__inform__type_1aab6f168571c2073e01e240524b8a3da0>`,
		:ref:`iter_pass2<doxid-structlstr__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4>`,
		:ref:`biters<doxid-structlstr__inform__type_1a0c5347be8391fbb23d728cebe0f3a5a8>`,
		:ref:`biter_min<doxid-structlstr__inform__type_1a6fe473492218a28f33e53f014c741e81>`,
		:ref:`biter_max<doxid-structlstr__inform__type_1aaa032644e73bb5bbc6092733db7f013b>`,
		:ref:`multiplier<doxid-structlstr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`,
		:ref:`x_norm<doxid-structlstr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`,
		:ref:`r_norm<doxid-structlstr__inform__type_1ae908410fabf891cfd89626c3605c38ca>`,
		:ref:`Atr_norm<doxid-structlstr__inform__type_1a0dc3a69b13123a76ec6ee7dd031eadff>`,
		:ref:`biter_mean<doxid-structlstr__inform__type_1a0c9f077f6c3bc52c519c2045c0578b22>`,
	}

.. _details-structlstr__inform__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

inform derived type as a C struct

Fields
------

.. index:: pair: variable; status
.. _doxid-structlstr__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

return status. See :ref:`lstr_solve_problem <doxid-galahad__lstr_8h_1af3355e5a8df63a9c7173eb974a1e7562>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structlstr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlstr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlstr__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structlstr__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	iter_pass2

the total number of pass-2 iterations required if the solution lies on the trust-region boundary

.. index:: pair: variable; biters
.. _doxid-structlstr__inform__type_1a0c5347be8391fbb23d728cebe0f3a5a8:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	biters

the total number of inner iterations performed

.. index:: pair: variable; biter_min
.. _doxid-structlstr__inform__type_1a6fe473492218a28f33e53f014c741e81:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	biter_min

the smallest number of inner iterations performed during an outer iteration

.. index:: pair: variable; biter_max
.. _doxid-structlstr__inform__type_1aaa032644e73bb5bbc6092733db7f013b:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	biter_max

the largestt number of inner iterations performed during an outer iteration

.. index:: pair: variable; multiplier
.. _doxid-structlstr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	multiplier

the Lagrange multiplier, :math:`\lambda`, corresponding to the trust-region constraint

.. index:: pair: variable; x_norm
.. _doxid-structlstr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	x_norm

the Euclidean norm of :math:`x`

.. index:: pair: variable; r_norm
.. _doxid-structlstr__inform__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	r_norm

the Euclidean norm of :math:`Ax-b`

.. index:: pair: variable; Atr_norm
.. _doxid-structlstr__inform__type_1a0dc3a69b13123a76ec6ee7dd031eadff:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	Atr_norm

the Euclidean norm of :math:`A^T (Ax-b) + \lambda x`

.. index:: pair: variable; biter_mean
.. _doxid-structlstr__inform__type_1a0c9f077f6c3bc52c519c2045c0578b22:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	biter_mean

the average number of inner iterations performed during an outer

iteration

