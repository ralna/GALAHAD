.. index:: pair: struct; lsqp_inform_type
.. _doxid-structlsqp__inform__type:

lsqp_inform_type structure
--------------------------

.. toctree::
	:hidden:


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_lsqp.h>
	
	struct lsqp_inform_type {
		// components
	
		int :ref:`status<doxid-structlsqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		int :ref:`alloc_status<doxid-structlsqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structlsqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		int :ref:`iter<doxid-structlsqp__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		int :ref:`factorization_status<doxid-structlsqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210>`;
		int64_t :ref:`factorization_integer<doxid-structlsqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e>`;
		int64_t :ref:`factorization_real<doxid-structlsqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc>`;
		int :ref:`nfacts<doxid-structlsqp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f>`;
		int :ref:`nbacts<doxid-structlsqp__inform__type_1a4b9a11ae940f04846c342978808696d6>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`obj<doxid-structlsqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`potential<doxid-structlsqp__inform__type_1a85f37aa42c9e051ea61ae035ff63059e>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`non_negligible_pivot<doxid-structlsqp__inform__type_1a827ddb7fead8e375404c9b770b67e771>`;
		bool :ref:`feasible<doxid-structlsqp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321>`;
		struct :ref:`lsqp_time_type<doxid-structlsqp__time__type>` :ref:`time<doxid-structlsqp__inform__type_1aee156075a3b7db49b39ffc8b0c254d7a>`;
		struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` :ref:`fdc_inform<doxid-structlsqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4>`;
		struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` :ref:`sbls_inform<doxid-structlsqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68>`;
	};
.. _details-structlsqp__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlsqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int status

return status. See LSQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlsqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlsqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlsqp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structlsqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structlsqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structlsqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int64_t factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structlsqp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structlsqp__inform__type_1a4b9a11ae940f04846c342978808696d6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; obj
.. _doxid-structlsqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` obj

the value of the objective function at the best estimate of the solution determined by LSQP_solve_qp

.. index:: pair: variable; potential
.. _doxid-structlsqp__inform__type_1a85f37aa42c9e051ea61ae035ff63059e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` potential

the value of the logarithmic potential function sum -log(distance to constraint boundary)

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structlsqp__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structlsqp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; time
.. _doxid-structlsqp__inform__type_1aee156075a3b7db49b39ffc8b0c254d7a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_time_type<doxid-structlsqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structlsqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structlsqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

