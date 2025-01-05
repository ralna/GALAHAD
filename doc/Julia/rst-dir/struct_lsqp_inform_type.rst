.. index:: pair: struct; lsqp_inform_type
.. _doxid-structlsqp__inform__type:

lsqp_inform_type structure
--------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct lsqp_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          nfacts::INT
          nbacts::INT
          obj::T
          potential::T
          non_negligible_pivot::T
          feasible::Bool
          time::lsqp_time_type{T}
          fdc_inform::fdc_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}

.. _details-structlsqp__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structlsqp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See LSQP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structlsqp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structlsqp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structlsqp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structlsqp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structlsqp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structlsqp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structlsqp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; nbacts
.. _doxid-structlsqp__inform__type_1a4b9a11ae940f04846c342978808696d6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nbacts

the total number of "wasted" function evaluations during the linesearch

.. index:: pair: variable; obj
.. _doxid-structlsqp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by LSQP_solve_qp

.. index:: pair: variable; potential
.. _doxid-structlsqp__inform__type_1a85f37aa42c9e051ea61ae035ff63059e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T potential

the value of the logarithmic potential function sum -log(distance to constraint boundary)

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structlsqp__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structlsqp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned "solution" feasible?

.. index:: pair: variable; time
.. _doxid-structlsqp__inform__type_1aee156075a3b7db49b39ffc8b0c254d7a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`lsqp_time_type<doxid-structlsqp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structlsqp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structlsqp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS

