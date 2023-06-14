.. index:: pair: table; glrt_inform_type
.. _doxid-structglrt__inform__type:

glrt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <galahad_glrt.h>
	
	struct glrt_inform_type {
		// fields
	
		int :ref:`status<doxid-structglrt__inform__type_1a6e27f49150e9a14580fb313cc2777e00>`;
		int :ref:`alloc_status<doxid-structglrt__inform__type_1a4335d5f44067aca76d5fff71eeb7d381>`;
		char :ref:`bad_alloc<doxid-structglrt__inform__type_1a19ba64e8444ca3672abd157e4f1303a3>`[81];
		int :ref:`iter<doxid-structglrt__inform__type_1aab6f168571c2073e01e240524b8a3da0>`;
		int :ref:`iter_pass2<doxid-structglrt__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`obj<doxid-structglrt__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`obj_regularized<doxid-structglrt__inform__type_1a1631e243108715d623e2ddb83310fa33>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`multiplier<doxid-structglrt__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`xpo_norm<doxid-structglrt__inform__type_1a145ebf82ab029a86c0bd00aec2ee4ae0>`;
		:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` :ref:`leftmost<doxid-structglrt__inform__type_1ab90b7ed1b1bfb32aeba7ad89a9a706d0>`;
		bool :ref:`negative_curvature<doxid-structglrt__inform__type_1aee928a2d12ccd5c99a5f3e65e9926021>`;
		bool :ref:`hard_case<doxid-structglrt__inform__type_1a22215075b7081ccac9f121daf07a0f7e>`;
	};
.. _details-structglrt__inform__type:

detailed documentation
----------------------

inform derived type as a C struct

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structglrt__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int status

return status. See :ref:`glrt_solve_problem <doxid-galahad__glrt_8h_1aa5e9905bd3a79584bc5133b7f7a6816f>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structglrt__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structglrt__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	char bad_alloc[81]

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structglrt__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structglrt__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; obj
.. _doxid-structglrt__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` obj

the value of the quadratic function

.. index:: pair: variable; obj_regularized
.. _doxid-structglrt__inform__type_1a1631e243108715d623e2ddb83310fa33:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` obj_regularized

the value of the regularized quadratic function

.. index:: pair: variable; multiplier
.. _doxid-structglrt__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` multiplier

the multiplier, :math:`\sigma \|x\|^{p-2}`

.. index:: pair: variable; xpo_norm
.. _doxid-structglrt__inform__type_1a145ebf82ab029a86c0bd00aec2ee4ae0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` xpo_norm

the value of the norm :math:`\|x\|_M`

.. index:: pair: variable; leftmost
.. _doxid-structglrt__inform__type_1ab90b7ed1b1bfb32aeba7ad89a9a706d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`real_wp_<doxid-galahad__precision_8h_1ab82133d435678ff159433d2e50cf295e>` leftmost

an estimate of the leftmost generalized eigenvalue of the pencil :math:`(H,M)`

.. index:: pair: variable; negative_curvature
.. _doxid-structglrt__inform__type_1aee928a2d12ccd5c99a5f3e65e9926021:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool negative_curvature

was negative curvature encountered ?

.. index:: pair: variable; hard_case
.. _doxid-structglrt__inform__type_1a22215075b7081ccac9f121daf07a0f7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool hard_case

did the hard case occur ?

