.. index:: pair: struct; wcp_inform_type
.. _doxid-structwcp__inform__type:

wcp_inform_type structure
-------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct wcp_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          iter::INT
          factorization_status::INT
          factorization_integer::Int64
          factorization_real::Int64
          nfacts::INT
          c_implicit::INT
          x_implicit::INT
          y_implicit::INT
          z_implicit::INT
          obj::T
          mu_final_target_max::T
          non_negligible_pivot::T
          feasible::Bool
          time::wcp_time_type{T}
          fdc_inform::fdc_inform_type{T,INT}
          sbls_inform::sbls_inform_type{T,INT}

.. _details-structwcp__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structwcp__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

return status. See WCP_solve for details

.. index:: pair: variable; alloc_status
.. _doxid-structwcp__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

the status of the last attempted allocation/deallocation

.. index:: pair: variable; bad_alloc
.. _doxid-structwcp__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

the name of the array for which an allocation/deallocation error occurred

.. index:: pair: variable; iter
.. _doxid-structwcp__inform__type_1aab6f168571c2073e01e240524b8a3da0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iter

the total number of iterations required

.. index:: pair: variable; factorization_status
.. _doxid-structwcp__inform__type_1aa448fed9eb03e70d5a03300b4fbbf210:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT factorization_status

the return status from the factorization

.. index:: pair: variable; factorization_integer
.. _doxid-structwcp__inform__type_1a29cd3a5b0f30227170f825116d9ade9e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_integer

the total integer workspace required for the factorization

.. index:: pair: variable; factorization_real
.. _doxid-structwcp__inform__type_1ad73643c24d3cd34c356c3ccd2ebfb1cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 factorization_real

the total real workspace required for the factorization

.. index:: pair: variable; nfacts
.. _doxid-structwcp__inform__type_1af54a1b17cb663c1e89a5bcd5f1e9961f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nfacts

the total number of factorizations performed

.. index:: pair: variable; c_implicit
.. _doxid-structwcp__inform__type_1a67593aebe8ddd8c5a8d66377bd1eaf00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT c_implicit

the number of general constraints that lie on (one) of their bounds for feasible solutions

.. index:: pair: variable; x_implicit
.. _doxid-structwcp__inform__type_1a15ecbcc95a8f49b406624abba6f9efe1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT x_implicit

the number of variables that lie on (one) of their bounds for all feasible solutions

.. index:: pair: variable; y_implicit
.. _doxid-structwcp__inform__type_1a7164fa79af695ec1d80d860366b291c9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT y_implicit

the number of Lagrange multipliers for general constraints that lie on (one) of their bounds for all feasible solutions

.. index:: pair: variable; z_implicit
.. _doxid-structwcp__inform__type_1a52829cf481145b7ec90572748e73331b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT z_implicit

the number of dual variables that lie on (one) of their bounds for all feasible solutions

.. index:: pair: variable; obj
.. _doxid-structwcp__inform__type_1a0cbcb28977ac1f47ab67d27e4216626d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T obj

the value of the objective function at the best estimate of the solution determined by WCP_solve

.. index:: pair: variable; mu_final_target_max
.. _doxid-structwcp__inform__type_1accc994a96bff755fa1ca935daf9ec4d8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mu_final_target_max

the largest target value on termination

.. index:: pair: variable; non_negligible_pivot
.. _doxid-structwcp__inform__type_1a827ddb7fead8e375404c9b770b67e771:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T non_negligible_pivot

the smallest pivot which was not judged to be zero when detecting linear dependent constraints

.. index:: pair: variable; feasible
.. _doxid-structwcp__inform__type_1aa43a71eb35dd7b8676c0b6236ceee321:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool feasible

is the returned primal-dual "solution" strictly feasible?

.. index:: pair: variable; time
.. _doxid-structwcp__inform__type_1afd2e59f6d17df493c93987f3f6b8b042:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	wcp_time_type{T} :ref:`wcp_time_type<doxid-structwcp__time__type>` time

timings (see above)

.. index:: pair: variable; fdc_inform
.. _doxid-structwcp__inform__type_1a966b6933e7b53fb2d71f55f267ad00f4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	fdc_inform_type{T,INT} :ref:`fdc_inform_type<doxid-structfdc__inform__type>` fdc_inform

inform parameters for FDC

.. index:: pair: variable; sbls_inform
.. _doxid-structwcp__inform__type_1a7e7617645ca9908f4f75e5216bb7cf68:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	sbls_inform_type{T,INT} :ref:`sbls_inform_type<doxid-structsbls__inform__type>` sbls_inform

inform parameters for SBLS
