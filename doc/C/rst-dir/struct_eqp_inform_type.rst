.. index:: pair: table; eqp_inform_type
.. _doxid-structeqp__inform__type:

eqp_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: lua
	:class: doxyrest-overview-code-block

	eqp_inform_type = {
		-- fields
	
		:ref:`status<doxid-structeqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`,
		:ref:`alloc_status<doxid-structeqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`,
		:ref:`bad_alloc<doxid-structeqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`,
		:ref:`cg_iter<doxid-structeqp__inform__type_1ad37cf7ad93af3413bc01b6515aad692a>`,
		:ref:`cg_iter_inter<doxid-structeqp__inform__type_1af9cff1fabd7b996847d1c93490c8db15>`,
		:ref:`factorization_integer<doxid-structeqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`,
		:ref:`factorization_real<doxid-structeqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`,
		:ref:`obj<doxid-structeqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`,
		:ref:`time<doxid-structeqp__inform__type_1ab4ea6394e359e4f2ba2543eda324643a>`,
		:ref:`fdc_inform<doxid-structeqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4>`,
		:ref:`sbls_inform<doxid-structeqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`,
		:ref:`gltr_inform<doxid-structeqp__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46>`,
	}

.. _details-structeqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structeqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	status

return status. See EQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structeqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structeqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; cg_iter
.. _doxid-structeqp__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	cg_iter

the total number of conjugate gradient iterations required

.. index:: pair: variable; cg_iter_inter
.. _doxid-structeqp__inform__type_1af9cff1fabd7b996847d1c93490c8db15:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	cg_iter_inter

see cg_iter

.. index:: pair: variable; factorization_integer
.. _doxid-structeqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structeqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; obj
.. _doxid-structeqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	obj

the value of the objective function at the best estimate of the solution determined by QPB_solve

.. index:: pair: variable; time
.. _doxid-structeqp__inform__type_1ab4ea6394e359e4f2ba2543eda324643a:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structeqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structeqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	sbls_inform

inform parameters for SBLS

.. index:: pair: variable; gltr_inform
.. _doxid-structeqp__inform__type_1a27a98844f05f18669d3dd60d3e6a8e46:

.. ref-code-block:: lua
	:class: doxyrest-title-code-block

	gltr_inform

return information from GLTR

