.. index:: pair: table; sls_inform_type
.. _doxid-structsls__inform__type:

sls_inform_type structure
-------------------------

.. toctree::
	:hidden:

	struct_hsl_types.rst
	struct_ssids_types.rst

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct sls_inform_type{T,INT}
          status::INT
          alloc_status::INT
          bad_alloc::NTuple{81,Cchar}
          more_info::INT
          entries::INT
          out_of_range::INT
          duplicates::INT
          upper::INT
          missing_diagonals::INT
          max_depth_assembly_tree::INT
          nodes_assembly_tree::INT
          real_size_desirable::Int64
          integer_size_desirable::Int64
          real_size_necessary::Int64
          integer_size_necessary::Int64
          real_size_factors::Int64
          integer_size_factors::Int64
          entries_in_factors::Int64
          max_task_pool_size::INT
          max_front_size::INT
          compresses_real::INT
          compresses_integer::INT
          two_by_two_pivots::INT
          semi_bandwidth::INT
          delayed_pivots::INT
          pivot_sign_changes::INT
          static_pivots::INT
          first_modified_pivot::INT
          rank::INT
          negative_eigenvalues::INT
          num_zero::INT
          iterative_refinements::INT
          flops_assembly::Int64
          flops_elimination::Int64
          flops_blas::Int64
          largest_modified_pivot::T
          minimum_scaling_factor::T
          maximum_scaling_factor::T
          condition_number_1::T
          condition_number_2::T
          backward_error_1::T
          backward_error_2::T
          forward_error::T
          alternative::Bool
          solver::NTuple{21,Cchar}
          time::sls_time_type{T}
          sils_ainfo::sils_ainfo_type{T,INT}
          sils_finfo::sils_finfo_type{T,INT}
          sils_sinfo::sils_sinfo_type{T,INT}
          ma57_ainfo::ma57_ainfo{T,INT}
          ma57_finfo::ma57_finfo{T,INT}
          ma57_sinfo::ma57_sinfo{T,INT}
          ma77_info::ma77_info{T,INT}
          ma86_info::ma86_info{T,INT}
          ma87_info::ma87_info{T,INT}
          ma97_info::ma97_info{T,INT}
          ssids_inform::spral_ssids_inform{INT}
          mc61_info::NTuple{10,INT}
          mc61_rinfo::NTuple{15,T}
          mc64_info::mc64_info{INT}
          mc68_info::mc68_info{INT}
          mc77_info::NTuple{10,INT}
          mc77_rinfo::NTuple{10,T}
          mumps_error::INT
          mumps_info::NTuple{80,INT}
          mumps_rinfo::NTuple{40,T}
          pardiso_error::INT
          pardiso_IPARM::NTuple{64,INT}
          pardiso_DPARM::NTuple{64,T}
          mkl_pardiso_error::INT
          mkl_pardiso_IPARM::NTuple{64,INT}
          pastix_info::INT
          wsmp_error::INT
          wsmp_iparm::NTuple{64,INT}
          wsmp_dparm::NTuple{64,T}
          mpi_ierr::INT
          lapack_error::INT

.. _details-structsls__inform__type:

detailed documentation
----------------------

inform derived type as a Julia structure

components
~~~~~~~~~~

.. index:: pair: variable; status
.. _doxid-structsls__inform__type_1a6e27f49150e9a14580fb313cc2777e00:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT status

reported return status. Possible values are:

* **0**

  success 

* **-1**

  allocation error 

* **-2** 

  deallocation error 

* **-3**

  matrix data faulty (n < 1, ne < 0) 

* **-20**

  alegedly postive definite definite matrix is indefinite

* **-29**

  unavailable option 

* **-31**

  input order is not a permutation or is faulty in some other way 

* **-32**

  > control.max_integer_factor_size integer space required for factor 

* **-33**

  > control.max_real_factor_size real space required for factors 

* **-40**

  not possible to alter the diagonals 

* **-41**

  no access to permutation or pivot sequence used 

* **-42**

  no access to diagonal perturbations 

* **-43**

  direct-access file error 

* **-50**

  solver-specific error; see the solver's info parameter 

* **-101**

  unknown solver

.. index:: pair: variable; alloc_status
.. _doxid-structsls__inform__type_1a4335d5f44067aca76d5fff71eeb7d381:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT alloc_status

STAT value after allocate failure.

.. index:: pair: variable; bad_alloc
.. _doxid-structsls__inform__type_1a19ba64e8444ca3672abd157e4f1303a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	NTuple{81,Cchar} bad_alloc

name of array which provoked an allocate failure

.. index:: pair: variable; more_info
.. _doxid-structsls__inform__type_1a24d9e61a5ee1839c2fda4f8d5cff0cb7:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT more_info

further information on failure

.. index:: pair: variable; entries
.. _doxid-structsls__inform__type_1a05de9d2e9fdfcc8bf932ca13b95ede29:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT entries

number of entries

.. index:: pair: variable; out_of_range
.. _doxid-structsls__inform__type_1a8daa2a776cae6116e9f14e2b009430a5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT out_of_range

number of indices out-of-range

.. index:: pair: variable; duplicates
.. _doxid-structsls__inform__type_1a4266bf48aafe2914b08e60d6ef9cf446:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT duplicates

number of duplicates

.. index:: pair: variable; upper
.. _doxid-structsls__inform__type_1a0a1a19aadb8cf4f2b05d37a8798b667c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT upper

number of entries from the strict upper triangle

.. index:: pair: variable; missing_diagonals
.. _doxid-structsls__inform__type_1a8d33160feb6e388439a1d38641b00b3d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT missing_diagonals

number of missing diagonal entries for an allegedly-definite matrix

.. index:: pair: variable; max_depth_assembly_tree
.. _doxid-structsls__inform__type_1a6d6dc5b49ec465fb45c83d1a0de40e02:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_depth_assembly_tree

maximum depth of the assembly tree

.. index:: pair: variable; nodes_assembly_tree
.. _doxid-structsls__inform__type_1ac0b9b02071c45a104c814d8664d56a0c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT nodes_assembly_tree

nodes in the assembly tree (= number of elimination steps)

.. index:: pair: variable; real_size_desirable
.. _doxid-structsls__inform__type_1a578f47757db464cbf8ed9b83bc0fe19b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 real_size_desirable

desirable or actual size for real array for the factors and other data

.. index:: pair: variable; integer_size_desirable
.. _doxid-structsls__inform__type_1a8dd3cb87057fef593aa3a13a5b537e03:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 integer_size_desirable

desirable or actual size for integer array for the factors and other dat

.. index:: pair: variable; real_size_necessary
.. _doxid-structsls__inform__type_1a81e92687a9d89d7449fe0c7e108c99e5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 real_size_necessary

necessary size for real array for the factors and other data

.. index:: pair: variable; integer_size_necessary
.. _doxid-structsls__inform__type_1a8ac7a09dd6adc2d2a535e7a14a43c8d3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 integer_size_necessary

necessary size for integer array for the factors and other data

.. index:: pair: variable; real_size_factors
.. _doxid-structsls__inform__type_1a2fb34bb982d673eade451dd5044b5ed0:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 real_size_factors

predicted or actual number of reals to hold factors

.. index:: pair: variable; integer_size_factors
.. _doxid-structsls__inform__type_1aa0143d6a3ae9a2a606b8631ad3da7610:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 integer_size_factors

predicted or actual number of integers to hold factors

.. index:: pair: variable; entries_in_factors
.. _doxid-structsls__inform__type_1ab741fb84b79d2b013a84d71932452681:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 entries_in_factors

number of entries in factors

.. index:: pair: variable; max_task_pool_size
.. _doxid-structsls__inform__type_1adbfc63b37c8305f76154af9ab21a73e1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_task_pool_size

maximum number of tasks in the factorization task pool

.. index:: pair: variable; max_front_size
.. _doxid-structsls__inform__type_1a854d688fc9f75d1011046b68798c3ec3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT max_front_size

forecast or actual size of largest front

.. index:: pair: variable; compresses_real
.. _doxid-structsls__inform__type_1a08f180b15976fad5be1793aeadd89d1d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT compresses_real

number of compresses of real data

.. index:: pair: variable; compresses_integer
.. _doxid-structsls__inform__type_1ae278663a9e98a86e0947f89a75549d51:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT compresses_integer

number of compresses of integer data

.. index:: pair: variable; two_by_two_pivots
.. _doxid-structsls__inform__type_1af922a8827cb6bb34ae8b7ea113eb88d9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT two_by_two_pivots

number of 2x2 pivots

.. index:: pair: variable; semi_bandwidth
.. _doxid-structsls__inform__type_1abf884043df0f9c0d95bcff6fae1bf9bb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT semi_bandwidth

semi-bandwidth of matrix following bandwidth reduction

.. index:: pair: variable; delayed_pivots
.. _doxid-structsls__inform__type_1aceacc7f8144f94b9ff16506b1bc581e3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT delayed_pivots

number of delayed pivots (total)

.. index:: pair: variable; pivot_sign_changes
.. _doxid-structsls__inform__type_1ac41ffa6bf566e674ac90de4178c81477:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pivot_sign_changes

number of pivot sign changes if no pivoting is used successfully

.. index:: pair: variable; static_pivots
.. _doxid-structsls__inform__type_1ab7569a21ad5852708f0a43b7c269ac0d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT static_pivots

number of static pivots chosen

.. index:: pair: variable; first_modified_pivot
.. _doxid-structsls__inform__type_1a735be3d14946330c6c7b451743b86ae9:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT first_modified_pivot

first pivot modification when static pivoting

.. index:: pair: variable; rank
.. _doxid-structsls__inform__type_1a6cfd95afd0afebd625b889fb6e58371c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT rank

estimated rank of the matrix

.. index:: pair: variable; negative_eigenvalues
.. _doxid-structsls__inform__type_1ad9bf58dadb2657173be0588308d100b3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT negative_eigenvalues

number of negative eigenvalues

.. index:: pair: variable; num_zero
.. _doxid-structsls__inform__type_1a9e21341bbd4963be2eaae5a0f8851648:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT num_zero

number of pivots that are considered zero (and ignored)

.. index:: pair: variable; iterative_refinements
.. _doxid-structsls__inform__type_1a22c83a8ec8964a169bdd9d0cdf196cf1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT iterative_refinements

number of iterative refinements performed

.. index:: pair: variable; flops_assembly
.. _doxid-structsls__inform__type_1a24b90be205ee95fe42ce3dc31d77ccfa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 flops_assembly

anticipated or actual number of floating-point operations in assembly

.. index:: pair: variable; flops_elimination
.. _doxid-structsls__inform__type_1a9dde906a7f13e691e9a885e827179c8d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 flops_elimination

anticipated or actual number of floating-point operations in elimination

.. index:: pair: variable; flops_blas
.. _doxid-structsls__inform__type_1a34f890f9248fc1f05ba5fae571990d6d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Int64 flops_blas

additional number of floating-point operations for BLAS

.. index:: pair: variable; largest_modified_pivot
.. _doxid-structsls__inform__type_1a62d158ab4a7acedf5d7a06841765ec49:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T largest_modified_pivot

largest diagonal modification when static pivoting or ensuring definiten

.. index:: pair: variable; minimum_scaling_factor
.. _doxid-structsls__inform__type_1a250fb521c37638fcca8c2867ff43b576:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T minimum_scaling_factor

minimum scaling factor

.. index:: pair: variable; maximum_scaling_factor
.. _doxid-structsls__inform__type_1ad8f20093907b8b8d244b4e37a13143cc:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T maximum_scaling_factor

maximum scaling factor

.. index:: pair: variable; condition_number_1
.. _doxid-structsls__inform__type_1a198ba7b98aa903f1ab5b65745c5289ea:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T condition_number_1

esimate of the condition number of the matrix (category 1 equations)

.. index:: pair: variable; condition_number_2
.. _doxid-structsls__inform__type_1a9ffbf5df4bb4e49067cf2813ad6a016e:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T condition_number_2

estimate of the condition number of the matrix (category 2 equations)

.. index:: pair: variable; backward_error_1
.. _doxid-structsls__inform__type_1a83c0d093c0b2b8636f686ab30acfb5a3:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T backward_error_1

esimate of the backward error (category 1 equations)

.. index:: pair: variable; backward_error_2
.. _doxid-structsls__inform__type_1a6fab94020d1a5e4b25a5006103a231cf:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T backward_error_2

esimate of the backward error (category 2 equations)

.. index:: pair: variable; forward_error
.. _doxid-structsls__inform__type_1ab63fd70aff6616a414f16af190bd4105:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T forward_error

estimate of forward error

.. index:: pair: variable; alternative
.. _doxid-structsls__inform__type_1a48c07c7da1803ed8af25ca949f4854b5:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	Bool alternative

has an "alternative" $y$: $A y = 0$ and $y^T b > 0$ been found when trying 
to solve $A x = b$ ?

.. index:: pair: variable; solver
.. _doxid-structsls__inform__type_1af335c33211ea78f2d5b7314d7b1c210d:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	char solver[21]

name of external solver used to factorize and solve

.. index:: pair: variable; time
.. _doxid-structsls__inform__type_1aeb306e2c697a3b16156c7bbf95933d04:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sls_time_type<doxid-structsls__time__type>` time

timings (see above)

.. index:: pair: variable; sils_ainfo
.. _doxid-structsls__inform__type_1a6490df077bf8d853c6ce6242ef95ef3f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sils_ainfo_type<doxid-structsils__ainfo__type>` sils_ainfo

the analyse output structure from sils

.. index:: pair: variable; sils_finfo
.. _doxid-structsls__inform__type_1aaca9130b6f951c48e420ef2d880e2e6a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sils_finfo_type<doxid-structsils__finfo__type>` sils_finfo

the factorize output structure from sils

.. index:: pair: variable; sils_sinfo
.. _doxid-structsls__inform__type_1a02b2db43c1e7bc2d42a9f584433dbe01:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`sils_sinfo_type<doxid-structsils__sinfo__type>` sils_sinfo

the solve output structure from sils

.. index:: pair: variable; ma57_ainfo
.. _doxid-structsls__inform__type_1aa7061ae61ef272126c2d93c0bb7c3e92:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma57_ainfo <details-structma57__ainfo>` ma57_ainfo

the analyse output structure from hsl_ma57

.. index:: pair: variable; ma57_finfo
.. _doxid-structsls__inform__type_1ae25ef469002f1c7bdbfcb22780e58d6f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma57_finfo <details-structma57__finfo>` ma57_finfo

the factorize output structure from hsl_ma57

.. index:: pair: variable; ma57_sinfo
.. _doxid-structsls__inform__type_1a87ebdff77dff15b8a72f8fb8528495f8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma57_sinfo <details-structma57__sinfo>` ma57_sinfo

the solve output structure from hsl_ma57

.. index:: pair: variable; ma77_info
.. _doxid-structsls__inform__type_1a296c08f6844db0dd56124633c5bead05:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma77_info <details-structma77__info>` ma77_info

the output structure from hsl_ma77

.. index:: pair: variable; ma86_info
.. _doxid-structsls__inform__type_1a0e8dbbed5f52f17a0c85cc24c1d4133a:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma86_info <details-structma86__info>` ma86_info

the output structure from hsl_ma86

.. index:: pair: variable; ma87_info
.. _doxid-structsls__inform__type_1a5a0d5c7fc9b533354415c7f26c5ca0fa:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma87_info <details-structma87__info>` ma87_info

the output structure from hsl_ma87

.. index:: pair: variable; ma97_info
.. _doxid-structsls__inform__type_1a97187c145bd03b5812c1b08cfd0c1fdd:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`ma97_info <details-structma97__info>` ma97_info

the output structure from hsl_ma97

.. index:: pair: variable; ssids_inform
.. _doxid-structsls__inform__type_1a1460057cd78a3850b14e78583b051054:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`spral_ssids_inform <details-structspral__ssids__inform>` ssids_inform

the output structure from ssids

.. index:: pair: variable; mc61_info
.. _doxid-structsls__inform__type_1ad29411cd0e18c59e43b474314a2adbe8:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mc61_info[10]

the real output array from mc61 from HSL

.. index:: pair: variable; mc61_rinfo
.. _doxid-structsls__inform__type_1a6966776cf11a3b9c447f7a1c9621152f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mc61_rinfo[15]

the integer output array from mc61 from HSL

.. index:: pair: variable; mc64_info
.. _doxid-structsls__inform__type_1a4b3b5b1c3585cea6e9006dcb9f34182c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct  :ref:`mc64_info <details-structmc64__info>` mc64_info

the output structure from hsl_mc64

.. index:: pair: variable; mc68_info
.. _doxid-structsls__inform__type_1a47a1ee8b0f64c664c47bbd23bacb4af6:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	struct :ref:`mc68_info <details-structmc68__info>` mc68_info

the output structure from hsl_mc68

.. index:: pair: variable; mc77_info
.. _doxid-structsls__inform__type_1ac143b7dbde27738930a06678d7a84135:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mc77_info[10]

the integer output array from mc77

.. index:: pair: variable; mc77_rinfo
.. _doxid-structsls__inform__type_1a02ac14c9a03a409b25bf374862f17166:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mc77_rinfo[10]

the real output array from mc77 from HSL

.. index:: pair: variable; mumps_error
.. _doxid-structsls__inform__type_1aabe7dce2361151a723670484e385d4e2:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mumps_error

the output scalars and arrays from mumps

.. index:: pair: variable; mumps_info
.. _doxid-structsls__inform__type_1a1e9b1230547090c9405ed025acddf937:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mumps_info[80]

see mumps_error

.. index:: pair: variable; mumps_rinfo
.. _doxid-structsls__inform__type_1a0e35523297d295b968ff61aff25c6279:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T mumps_rinfo[40]

see mumps_error

.. index:: pair: variable; pardiso_error
.. _doxid-structsls__inform__type_1af4edf991dc128b2aa6bb7c5909bce7bb:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pardiso_error

the output scalars and arrays from pardiso

.. index:: pair: variable; pardiso_IPARM
.. _doxid-structsls__inform__type_1a94bd760acd69f7e452771c96f9a6d83b:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pardiso_IPARM[64]

see pardiso_error

.. index:: pair: variable; pardiso_DPARM
.. _doxid-structsls__inform__type_1ad94ecfb58762fd0f79d1dcc51c1a6c13:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T pardiso_DPARM[64]

see pardiso_error

.. index:: pair: variable; mkl_pardiso_error
.. _doxid-structsls__inform__type_1a0af76cb45fcf59f7749bf196c1f848f1:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mkl_pardiso_error

the output scalars and arrays from mkl_pardiso

.. index:: pair: variable; mkl_pardiso_IPARM
.. _doxid-structsls__inform__type_1aeaa2b4360e0796404f57ede659a0dc5f:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mkl_pardiso_IPARM[64]

see mkl_pardiso_error

.. index:: pair: variable; pastix_info
.. _doxid-structsls__inform__type_1a8317c10c22116e0cf019d281a7f07595:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT pastix_info

the output flag from pastix

.. index:: pair: variable; wsmp_error
.. _doxid-structsls__inform__type_1a12ce464e4b92fb3ce21377773cf801ef:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT wsmp_error

the output scalars and arrays from wsmp

.. index:: pair: variable; wsmp_iparm
.. _doxid-structsls__inform__type_1a9dafb01a424625a16f75ba37f96e6067:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT wsmp_iparm[64]

see wsmp_error

.. index:: pair: variable; wsmp_dparm
.. _doxid-structsls__inform__type_1acec90b9af7496ccc0a7fc9fc0dbfb49c:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	T wsmp_dparm[64]

see wsmp_error

.. index:: pair: variable; mpi_ierr
.. _doxid-structsls__inform__type_1ab885ec4629b2033ab17e5ca273739488:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT mpi_ierr

the output flag from MPI routines

.. index:: pair: variable; lapack_error
.. _doxid-structsls__inform__type_1a0319e2c4ee2d95fe244f92d276038bd4:

.. ref-code-block:: julia
	:class: doxyrest-title-code-block

	INT lapack_error

the output flag from LAPACK routines

