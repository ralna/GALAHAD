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

       struct spral_ssids_options{T,INT}
          array_base::INT
          print_level::INT
          unit_diagnostics::INT
          unit_error::INT
          unit_warning::INT
          ordering::INT
          nemin::INT
          ignore_numa::Bool
          use_gpu::Bool
          gpu_only::Bool
          min_gpu_work::Int64
          max_load_inbalance::Cfloat
          gpu_perf_coeff::Cfloat
          scaling::INT
          small_subtree_threshold::Int64
          cpu_block_size::INT
          action::Bool
          pivot_method::INT
          small::T
          u::T
          nstream::INT
          multiplier::T
          min_loadbalance::Cfloat
          failed_pivot_method::INT

.. _details-structspral__ssids__inform:

spral_ssids_inform structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct spral_ssids_inform{INT}
          flag::INT
          matrix_dup::INT
          matrix_missing_diag::INT
          matrix_outrange::INT
          matrix_rank::INT
          maxdepth::INT
          maxfront::INT
          num_delay::INT
          num_factor::Int64
          num_flops::Int64
          num_neg::INT
          num_sup::INT
          num_two::INT
          stat::INT
          cuda_error::INT
          cublas_error::INT
          not_first_pass::INT
          not_second_pass::INT
          nparts::INT
          cpu_flops::Int64
          gpu_flops::Int64


detailed documentation
----------------------

SSIDS package option and info derived types as Julia structures.
See `SPRAL-SSIDS <https://github.com/ralna/spral>`_ 
documentation for further details.
