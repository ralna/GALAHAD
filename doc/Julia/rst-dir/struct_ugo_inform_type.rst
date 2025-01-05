.. index:: pair: struct; ugo_inform_type
.. _doxid-structugo__inform__type:

ugo_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ugo_inform_type{T,INT}
          status::INT
          eval_status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          f_eval::INT
          g_eval::INT
          h_eval::INT
          time::ugo_time_type{T}

.. _details-structugo__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structugo__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See UGO_solve for details

.. index:: pair: variable; eval_status
.. _doxid-structugo__inform__type_1a7b88f94f39913a54e69ea23c12d3e4bd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT eval_status

evaluation status for reverse communication interface

.. index:: pair: variable; alloc_status
.. _doxid-structugo__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structugo__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar) bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structugo__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structugo__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structugo__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT g_eval

the total number of evaluations of the gradient of the objective function

.. index:: pair: variable; h_eval
.. _doxid-structugo__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; time
.. _doxid-structugo__inform__type_1a9113f4cf33f961d4a1a97455a054eac6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	ugo_time_type{T} :ref:`ugo_time_type<doxid-structugo__time__type>` time

timings (see above)
