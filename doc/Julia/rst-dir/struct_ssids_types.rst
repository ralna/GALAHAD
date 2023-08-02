.. index:: pair: struct; ssids_control
.. index:: pair: struct; ssids_inform
.. _doxid-structssids__controlinform:

.. _details-structspral__ssids__options:

spral_ssids_options structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

       struct spral_ssids_options{T}
          array_base::Int32
          print_level::Int32
          unit_diagnostics::Int32
          unit_error::Int32
          unit_warning::Int32
          ordering::Int32
          nemin::Int32
          ignore_numa::Bool
          use_gpu::Bool
          gpu_only::Bool
          min_gpu_work::Int64
          max_load_inbalance::Cfloat
          gpu_perf_coeff::Cfloat
          scaling::Int32
          small_subtree_threshold::Int64
          cpu_block_size::Int32
          action::Bool
          pivot_method::Int32
          small::T
          u::T
          nstream::Int32
          multiplier::T
          min_loadbalance::Cfloat
          failed_pivot_method::Int32

.. _details-structspral__ssids__inform:

spral_ssids_inform structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct spral_ssids_inform
          flag::Int32
          matrix_dup::Int32
          matrix_missing_diag::Int32
          matrix_outrange::Int32
          matrix_rank::Int32
          maxdepth::Int32
          maxfront::Int32
          num_delay::Int32
          num_factor::Int64
          num_flops::Int64
          num_neg::Int32
          num_sup::Int32
          num_two::Int32
          stat::Int32
          cuda_error::Int32
          cublas_error::Int32
          not_first_pass::Int32
          not_second_pass::Int32
          nparts::Int32
          cpu_flops::Int64
          gpu_flops::Int64


detailed documentation
----------------------

SSIDS package option and info derived types as Julia structures.
See `SPRAL-SSIDS <https://github.com/ralna/spral>`_ 
documentation for further details.
