.. index:: pair: table; lsrt_inform_type
.. _doxid-structlsrt__inform__type:

lsrt_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lsrt_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          iter_pass2::INT
          biters::INT
          biter_min::INT
          biter_max::INT
          obj::T
          multiplier::T
          x_norm::T
          r_norm::T
          Atr_norm::T
          biter_mean::T

.. _details-structlsrt__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlsrt__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See :ref:`lsrt_solve_problem <doxid-galahad__lsrt_8h_1aa1b3479d5f21fe373ef8948d55763992>` for details

.. index:: pair: variable; alloc_status
.. _doxid-structlsrt__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlsrt__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlsrt__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; iter_pass2
.. _doxid-structlsrt__inform__type_1aa69f8ea5f07782fd8ad0318f87202ac4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter_pass2

the total number of pass-2 iterations required

.. index:: pair: variable; biters
.. _doxid-structlsrt__inform__type_1a0c5347be8391fbb23d728cebe0f3a5a8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT biters

the total number of inner iterations performed

.. index:: pair: variable; biter_min
.. _doxid-structlsrt__inform__type_1a6fe473492218a28f33e53f014c741e81:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT biter_min

the smallest number of inner iterations performed during an outer iteration

.. index:: pair: variable; biter_max
.. _doxid-structlsrt__inform__type_1aaa032644e73bb5bbc6092733db7f013b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT biter_max

the largest number of inner iterations performed during an outer iteration

.. index:: pair: variable; obj
.. _doxid-structlsrt__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function

.. index:: pair: variable; multiplier
.. _doxid-structlsrt__inform__type_1ac8bfb1ed777319ef92b7039c66f9a9b0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T multiplier

the multiplier, $\lambda = sigma ||x||^(p-2)$

.. index:: pair: variable; x_norm
.. _doxid-structlsrt__inform__type_1a32b3ba51ed1b0d7941f34e736da26ae3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T x_norm

the Euclidean norm of $x$

.. index:: pair: variable; r_norm
.. _doxid-structlsrt__inform__type_1ae908410fabf891cfd89626c3605c38ca:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T r_norm

the Euclidean norm of $Ax-b$

.. index:: pair: variable; Atr_norm
.. _doxid-structlsrt__inform__type_1a0dc3a69b13123a76ec6ee7dd031eadff:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T Atr_norm

the Euclidean norm of $A^T (Ax-b) + \lambda x$

.. index:: pair: variable; biter_mean
.. _doxid-structlsrt__inform__type_1a0c9f077f6c3bc52c519c2045c0578b22:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T biter_mean

the average number of inner iterations performed during an outer iteration

