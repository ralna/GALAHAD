.. index:: pair: table; trs_inform_type
.. _doxid-structtrs__inform__type:

trs_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. _details-structtrs__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. ---------------------------------------------------------------------------
.. index:: pair: table; trs_inform_type
.. _doxid-structtrs__inform__type:

table trs_inform_type
=====================


.. toctree::
	:hidden:

Overview
~~~~~~~~

inform derived type as a C struct :ref:`More...<details-structtrs__inform__type>`

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	trs_inform_type = {
		-- fields
	
		:ref:`status<doxid-structtrs__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structtrs__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`factorizations<doxid-structtrs__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f>`,
		:ref:`max_entries_factors<doxid-structtrs__inform__type_1a177e429e737cfa2cd3df051a65fcfb68>`,
		:ref:`len_history<doxid-structtrs__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6>`,
		:ref:`obj<doxid-structtrs__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`,
		:ref:`x_norm<doxid-structtrs__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3>`,
		:ref:`multiplier<doxid-structtrs__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`,
		:ref:`pole<doxid-structtrs__inform__type_1ad2dc9016b1d2b00a970ec28129f7000d>`,
		:ref:`dense_factorization<doxid-structtrs__inform__type_1a107cef1ccaad53efc9d7a578d400f324>`,
		:ref:`hard_case<doxid-structtrs__inform__type_1a22215075b7081ccac9f121daf07a0f7e>`,
		:ref:`bad_alloc<doxid-structtrs__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`time<doxid-structtrs__inform__type_1a137801ab863218dd25dd377da6a6cbfb>`,
		:ref:`history<doxid-structtrs__inform__type_1adbea75a6746b7545b3c4dfcfc8780664>`,
		:ref:`sls_inform<doxid-structtrs__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0>`,
		:ref:`ir_inform<doxid-structtrs__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5>`,
	}

.. _details-structtrs__inform__type:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

inform derived type as a C struct

Fields
------

.. index:: pair: variable; status
.. _doxid-structtrs__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

reported return status:

* 0 the solution has been found

* -1 an array allocation has failed

* -2 an array deallocation has failed

* -3 n and/or Delta is not positive

* -9 the analysis phase of the factorization of :math:`H + \lambda M` failed

* -10 the factorization of :math:`H + \lambda M` failed

* -15 :math:`M` does not appear to be strictly diagonally dominant

* -16 ill-conditioning has prevented further progress

.. index:: pair: variable; alloc_status
.. _doxid-structtrs__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

STAT value after allocate failure.

.. index:: pair: variable; factorizations
.. _doxid-structtrs__inform__type_1a9a6a5a0de7d7a6048b4170a768c0c86f:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorizations

the number of factorizations performed

.. index:: pair: variable; max_entries_factors
.. _doxid-structtrs__inform__type_1a177e429e737cfa2cd3df051a65fcfb68:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; len_history
.. _doxid-structtrs__inform__type_1a2087c1ee7c5859aa738d2f07ba91b4a6:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	len_history

the number of :math:`(||x||_M,\lambda)` pairs in the history

.. index:: pair: variable; obj
.. _doxid-structtrs__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	obj

the value of the quadratic function

.. index:: pair: variable; x_norm
.. _doxid-structtrs__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	x_norm

the :math:`M` -norm of :math:`x`, :math:`||x||_M`

.. index:: pair: variable; multiplier
.. _doxid-structtrs__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	multiplier

the Lagrange multiplier corresponding to the trust-region constraint

.. index:: pair: variable; pole
.. _doxid-structtrs__inform__type_1ad2dc9016b1d2b00a970ec28129f7000d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	pole

a lower bound max :math:`(0,-\lambda_1)`, where :math:`\lambda_1` is the left-most eigenvalue of :math:`(H,M)`

.. index:: pair: variable; dense_factorization
.. _doxid-structtrs__inform__type_1a107cef1ccaad53efc9d7a578d400f324:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	dense_factorization

was a dense factorization used?

.. index:: pair: variable; hard_case
.. _doxid-structtrs__inform__type_1a22215075b7081ccac9f121daf07a0f7e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	hard_case

has the hard case occurred?

.. index:: pair: variable; bad_alloc
.. _doxid-structtrs__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

name of array that provoked an allocate failure

.. index:: pair: variable; time
.. _doxid-structtrs__inform__type_1a137801ab863218dd25dd377da6a6cbfb:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time

time information

.. index:: pair: variable; history
.. _doxid-structtrs__inform__type_1adbea75a6746b7545b3c4dfcfc8780664:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	history

history information

.. index:: pair: variable; sls_inform
.. _doxid-structtrs__inform__type_1a0a9d7a6860aca6894830ccaabe3ceac0:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sls_inform

cholesky information (see sls_c documentation)

.. index:: pair: variable; ir_inform
.. _doxid-structtrs__inform__type_1ae3db15e2ecf7454c4db293d5b30bc7f5:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	ir_inform

iterative_refinement information (see ir_c documentation)

