.. index:: pair: table; llsr_inform_type
.. _doxid-structllsr__inform__type:

llsr_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. _details-structllsr__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: table; llsr_inform_type
.. _doxid-structllsr__inform__type:

table llsr_inform_type
======================


.. toctree::
	:hidden:

Overview
~~~~~~~~

inform derived type as a C struct :ref:`More...<details-structllsr__inform__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	llsr_inform_type = {
		-- fields
	
		:ref:`status<doxid-structllsr__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structllsr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`factorizations<doxid-structllsr__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f>`,
		:ref:`len_history<doxid-structllsr__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6>`,
		:ref:`r_norm<doxid-structllsr__inform__type_1ae908410fabf891cfd89626c3605c38ca>`,
		:ref:`x_norm<doxid-structllsr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`,
		:ref:`multiplier<doxid-structllsr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`,
		:ref:`bad_alloc<doxid-structllsr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`time<doxid-structllsr__inform__type_1ace18e9a0877156e432cc23c7d5799dd6>`,
		:ref:`history<doxid-structllsr__inform__type_1a13047d24b0cf3469a41cc14c364d3587>`,
		:ref:`sbls_inform<doxid-structllsr__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`,
		:ref:`sls_inform<doxid-structllsr__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`,
		:ref:`ir_inform<doxid-structllsr__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5>`,
	}

.. _details-structllsr__inform__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

inform derived type as a C struct

Fields
------

.. index:: pair: variable; status
.. _doxid-structllsr__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

reported return status:

* 0 the solution has been found

* -1 an array allocation has failed

* -2 an array deallocation has failed

* -3 n and/or Delta is not positive

* -10 the factorization of :math:`K(\lambda)` failed

* -15 :math:`S` does not appear to be strictly diagonally dominant

* -16 ill-conditioning has prevented furthr progress

.. index:: pair: variable; alloc_status
.. _doxid-structllsr__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structllsr__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorizations

the number of factorizations performed

.. index:: pair: variable; len_history
.. _doxid-structllsr__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	len_history

the number of (:math:`\|x\|_S`, :math:`\lambda`) pairs in the history

.. index:: pair: variable; r_norm
.. _doxid-structllsr__inform__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	r_norm

corresponding value of the two-norm of the residual, :math:`\|A x(\lambda) - b\|`

.. index:: pair: variable; x_norm
.. _doxid-structllsr__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	x_norm

the S-norm of x, :math:`\|x\|_S`

.. index:: pair: variable; multiplier
.. _doxid-structllsr__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	multiplier

the multiplier corresponding to the regularization term

.. index:: pair: variable; bad_alloc
.. _doxid-structllsr__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structllsr__inform__type_1ace18e9a0877156e432cc23c7d5799dd6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time

time information

.. index:: pair: variable; history
.. _doxid-structllsr__inform__type_1a13047d24b0cf3469a41cc14c364d3587:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	history

history information

.. index:: pair: variable; sbls_inform
.. _doxid-structllsr__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sbls_inform

information from the symmetric factorization and related linear solves (see sbls_c documentation)

.. index:: pair: variable; sls_inform
.. _doxid-structllsr__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sls_inform

information from the factorization of S and related linear solves (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structllsr__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	ir_inform

information from the iterative refinement for definite system solves (see ir_c documentation)

