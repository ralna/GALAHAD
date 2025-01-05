.. index:: pair: struct; arc_inform_type
.. _doxid-structarc__inform__type:

arc_inform_type structure
-------------------------

.. toctree::
	:hidden:


.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct arc_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          cg_iter::INT
          f_eval::INT
          g_eval::INT
          h_eval::INT
          factorization_status::INT
          factorization_max::INT
          max_entries_factors::Int64
          factorization_integer::Int64
          factorization_real::Int64
          factorization_average::T
          obj::T
          norm_g::T
          weight::T
          time::arc_time_type{T}
          rqs_inform::rqs_inform_type{T,INT}
          glrt_inform::glrt_inform_type{T,INT}
          dps_inform::dps_inform_type{T,INT}
          psls_inform::psls_inform_type{T,INT}
          lms_inform::lms_inform_type{T,INT}
          lms_inform_prec::lms_inform_type{T,INT}
          sha_inform::sha_inform_type{T,INT}
	
.. _details-structarc__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structarc__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See ARC_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structarc__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structarc__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structarc__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations performed

.. index:: pair: variable; cg_iter
.. _doxid-structarc__inform__type_1ad37cf7ad93af3413bc01b6515aad692a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT cg_iter

the total number of CG iterations performed

.. index:: pair: variable; f_eval
.. _doxid-structarc__inform__type_1aa9c29d7119d66d8540900c7531b2dcfa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT f_eval

the total number of evaluations of the objective function

.. index:: pair: variable; g_eval
.. _doxid-structarc__inform__type_1acd459eb95ff0f2d74e9cc3931d8e5469:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT g_eval

the total number of evaluations of the gradient of the objective functio

.. index:: pair: variable; h_eval
.. _doxid-structarc__inform__type_1af1410cb1718f2a083dd8a7dee9ab643a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT h_eval

the total number of evaluations of the Hessian of the objective function

.. index:: pair: variable; factorization_status
.. _doxid-structarc__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_max
.. _doxid-structarc__inform__type_1a97dadabf3b7bdf921c4dcd1f43129f05:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_max

the maximum number of factorizations in a sub-problem solve

.. index:: pair: variable; max_entries_factors
.. _doxid-structarc__inform__type_1a177e429e737cfa2cd3df051a65fcfb68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 max_entries_factors

the maximum number of entries in the factors

.. index:: pair: variable; factorization_integer
.. _doxid-structarc__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structarc__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; factorization_average
.. _doxid-structarc__inform__type_1a42d0c89df887685f68327d07c6e92f05:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T factorization_average

the average number of factorizations per sub-problem solve

.. index:: pair: variable; obj
.. _doxid-structarc__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by the package.

.. index:: pair: variable; norm_g
.. _doxid-structarc__inform__type_1ae1bc0a751c6ede62421bbc49fbe7d9fe:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T norm_g

the norm of the gradient of the objective function at the best estimate of the solution determined by the package.

.. index:: pair: variable; weight
.. _doxid-structarc__inform__type_1adcd20aeaf7042e972ddab56f3867ce70:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T weight

the current value of the regularization weight

.. index:: pair: variable; time
.. _doxid-structarc__inform__type_1a4288efaa60a28f967b947ef1b07cf445:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`arc_time_type<doxid-structarc__time__type>` time

timings (see above)

.. index:: pair: variable; rqs_inform
.. _doxid-structarc__inform__type_1a68497e7bbd1695ac9b830fc8fe594d60:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`rqs_inform_type<doxid-structrqs__inform__type>` rqs_inform

inform parameters for RQS

.. index:: pair: variable; glrt_inform
.. _doxid-structarc__inform__type_1aa5a47a840c1f9680ac8b9e4db3eb9e88:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`glrt_inform_type<doxid-structglrt__inform__type>` glrt_inform

inform parameters for GLRT

.. index:: pair: variable; dps_inform
.. _doxid-structarc__inform__type_1aec61ddb290b679c265693de171a6394f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`dps_inform_type<doxid-structdps__inform__type>` dps_inform

inform parameters for DPS

.. index:: pair: variable; psls_inform
.. _doxid-structarc__inform__type_1a57ca5ed37882eb917736f845d3cdb8ee:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`psls_inform_type<doxid-structpsls__inform__type>` psls_inform

inform parameters for PSLS

.. index:: pair: variable; lms_inform
.. _doxid-structarc__inform__type_1a6428cf213f8c899aa1bfb1fc3d24f37d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform

inform parameters for LMS

.. index:: pair: variable; lms_inform_prec
.. _doxid-structarc__inform__type_1a2040147e726e4ad18ef6d81d8339644e:
.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lms_inform_type<doxid-structlms__inform__type>` lms_inform_prec

inform parameters for LMS used for preconditioning

.. index:: pair: variable; sha_inform
.. _doxid-structarc__inform__type_1a196d9da91c7ed4a67aa6e009e336e101:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sha_inform_type<doxid-structsha__inform__type>` sha_inform

inform parameters for SHA

