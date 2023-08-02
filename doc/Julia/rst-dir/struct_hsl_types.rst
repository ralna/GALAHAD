.. index:: pair: struct; hsl_control
.. index:: pair: struct; hsl_info
.. _doxid-structhsl__controlinfo:

.. _details-structma48__control:

ma48_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_control{T}
          f_arrays::Int32
          multiplier::T
          u::T
          switch_::T
          drop::T
          tolerance::T
          cgce::T
          lp::Int32
          wp::Int32
          mp::Int32
          ldiag::Int32
          btf::Int32
          struct_::Int32
          maxit::Int32
          factor_blocking::Int32
          solve_blas::Int32
          pivoting::Int32
          diagonal_pivoting::Int32
          fill_in::Int32
          switch_mode::Int32

.. _details-structma48__ainfo:

ma48_ainfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_ainfo{T}
          ops::T
          flag::Int32
          more::Int32
          lena_analyse::Clong
          lenj_analyse::Clong
          lena_factorize::Clong
          leni_factorize::Clong
          ncmpa::Int32
          rank::Int32
          drop::Clong
          struc_rank::Int32
          oor::Clong
          dup::Clong
          stat::Int32
          lblock::Int32
          sblock::Int32
          tblock::Clong

.. _details-structma48__finfo:

ma48_finfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_finfo{T}
          ops::T
          flag::Int32
          more::Int32
          size_factor::Clong
          lena_factorize::Clong
          leni_factorize::Clong
          drop::Clong
          rank::Int32
          stat::Int32

.. _details-structma48__sinfo:

ma48_sinfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma48_sinfo
          flag::Int32
          more::Int32
          stat::Int32

.. _details-structma57__control:

ma57_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_control{T}
          f_arrays::Int32
          multiplier::T
          reduce::T
          u::T
          static_tolerance::T
          static_level::T
          tolerance::T
          convergence::T
          consist::T
          lp::Int32
          wp::Int32
          mp::Int32
          sp::Int32
          ldiag::Int32
          nemin::Int32
          factorblocking::Int32
          solveblocking::Int32
          la::Int32
          liw::Int32
          maxla::Int32
          maxliw::Int32
          pivoting::Int32
          thresh::Int32
          ordering::Int32
          scaling::Int32
          rank_deficient::Int32
          ispare::NTuple{5,Cint}
          rspare::NTuple{10,T}

.. _details-structma57__ainfo:

ma57_ainfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_ainfo{T}
          opsa::T
          opse::T
          flag::Int32
          more::Int32
          nsteps::Int32
          nrltot::Int32
          nirtot::Int32
          nrlnec::Int32
          nirnec::Int32
          nrladu::Int32
          niradu::Int32
          ncmpa::Int32
          ordering::Int32
          oor::Int32
          dup::Int32
          maxfrt::Int32
          stat::Int32
          ispare::NTuple{5,Cint}
          rspare::NTuple{10,T}

.. _details-structma57__finfo:

ma57_finfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_finfo{T}
          opsa::T
          opse::T
          opsb::T
          maxchange::T
          smin::T
          smax::T
          flag::Int32
          more::Int32
          maxfrt::Int32
          nebdu::Int32
          nrlbdu::Int32
          nirbdu::Int32
          nrltot::Int32
          nirtot::Int32
          nrlnec::Int32
          nirnec::Int32
          ncmpbr::Int32
          ncmpbi::Int32
          ntwo::Int32
          neig::Int32
          delay::Int32
          signc::Int32
          static_::Int32
          modstep::Int32
          rank::Int32
          stat::Int32
          ispare::NTuple{5,Cint}
          rspare::NTuple{10,T}

.. _details-structma57__sinfo:

ma57_sinfo structure
--------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma57_sinfo{T}
          cond::T
          cond2::T
          berr::T
          berr2::T
          error::T
          flag::Int32
          stat::Int32
          ispare::NTuple{5,Cint}
          rspare::NTuple{10,T}

.. _details-structma77__control:

ma77_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma77_control{T}
          f_arrays::Int32
          print_level::Int32
          unit_diagnostics::Int32
          unit_error::Int32
          unit_warning::Int32
          bits::Int32
          buffer_lpage::NTuple{2,Cint}
          buffer_npage::NTuple{2,Cint}
          file_size::Clong
          maxstore::Clong
          storage::NTuple{3,Clong}
          nemin::Int32
          maxit::Int32
          infnorm::Int32
          thresh::T
          nb54::Int32
          action::Int32
          multiplier::T
          nb64::Int32
          nbi::Int32
          small::T
          static_::T
          storage_indef::Clong
          u::T
          umin::T
          consist_tol::T
          ispare::NTuple{5,Cint}
          lspare::NTuple{5,Clong}
          rspare::NTuple{5,T}

.. _details-structma77__info:

ma77_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma77_info{T}
          detlog::T
          detsign::Int32
          flag::Int32
          iostat::Int32
          matrix_dup::Int32
          matrix_rank::Int32
          matrix_outrange::Int32
          maxdepth::Int32
          maxfront::Int32
          minstore::Clong
          ndelay::Int32
          nfactor::Clong
          nflops::Clong
          niter::Int32
          nsup::Int32
          num_neg::Int32
          num_nothresh::Int32
          num_perturbed::Int32
          ntwo::Int32
          stat::Int32
          index::NTuple{4,Cint}
          nio_read::NTuple{2,Clong}
          nio_write::NTuple{2,Clong}
          nwd_read::NTuple{2,Clong}
          nwd_write::NTuple{2,Clong}
          num_file::NTuple{4,Cint}
          storage::NTuple{4,Clong}
          tree_nodes::Int32
          unit_restart::Int32
          unused::Int32
          usmall::T
          ispare::NTuple{5,Cint}
          lspare::NTuple{5,Clong}
          rspare::NTuple{5,T}

.. _details-structma86__control:

ma86_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma86_control{T}
          f_arrays::Int32
          diagnostics_level::Int32
          unit_diagnostics::Int32
          unit_error::Int32
          unit_warning::Int32
          nemin::Int32
          nb::Int32
          action::Int32
          nbi::Int32
          pool_size::Int32
          small_::T
          static_::T
          u::T
          umin::T
          scaling::Int32

.. _details-structma86__info:

ma86_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma86_info{T}
          detlog::T
          detsign::Int32
          flag::Int32
          matrix_rank::Int32
          maxdepth::Int32
          num_delay::Int32
          num_factor::Clong
          num_flops::Clong
          num_neg::Int32
          num_nodes::Int32
          num_nothresh::Int32
          num_perturbed::Int32
          num_two::Int32
          pool_size::Int32
          stat::Int32
          usmall::T

.. _details-structma87__control:

ma87_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma87_control{T}
          f_arrays::Int32
          diagnostics_level::Int32
          unit_diagnostics::Int32
          unit_error::Int32
          unit_warning::Int32
          nemin::Int32
          nb::Int32
          pool_size::Int32
          diag_zero_minus::T
          diag_zero_plus::T
          unused::NTuple{40,Cchar}

.. _details-structma87__info:

ma87_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma87_info{T}
          detlog::T
          flag::Int32
          maxdepth::Int32
          num_factor::Clong
          num_flops::Clong
          num_nodes::Int32
          pool_size::Int32
          stat::Int32
          num_zero::Int32
          unused::NTuple{40,Cchar}

.. _details-structma97__control:

ma97_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma97_control{T}
          f_arrays::Int32
          action::Int32
          nemin::Int32
          multiplier::T
          ordering::Int32
          print_level::Int32
          scaling::Int32
          small::T
          u::T
          unit_diagnostics::Int32
          unit_error::Int32
          unit_warning::Int32
          factor_min::Clong
          solve_blas3::Int32
          solve_min::Clong
          solve_mf::Int32
          consist_tol::T
          ispare::NTuple{5,Cint}
          rspare::NTuple{10,T}

.. _details-structma97__info:

ma97_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct ma97_info{T}
          flag::Int32
          flag68::Int32
          flag77::Int32
          matrix_dup::Int32
          matrix_rank::Int32
          matrix_outrange::Int32
          matrix_missing_diag::Int32
          maxdepth::Int32
          maxfront::Int32
          num_delay::Int32
          num_factor::Clong
          num_flops::Clong
          num_neg::Int32
          num_sup::Int32
          num_two::Int32
          ordering::Int32
          stat::Int32
          ispare::NTuple{5,Cint}
          rspare::NTuple{10,T}

.. _details-structmc64__control:

mc64_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc64_control
          f_arrays::Int32
          lp::Int32
          wp::Int32
          sp::Int32
          ldiag::Int32
          checking::Int32

.. _details-structmc64__info:

mc64_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc64_info
          flag::Int32
          more::Int32
          strucrank::Int32
          stat::Int32

.. _details-structmc68__control:

mc68_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc68_control
          f_array_in::Int32
          f_array_out::Int32
          min_l_workspace::Int32
          lp::Int32
          wp::Int32
          mp::Int32
          nemin::Int32
          print_level::Int32
          row_full_thresh::Int32
          row_search::Int32

.. _details-structmc68__info:

mc68_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mc68_info
          flag::Int32
          iostat::Int32
          stat::Int32
          out_range::Int32
          duplicate::Int32
          n_compressions::Int32
          n_zero_eigs::Int32
          l_workspace::Clong
          zb01_info::Int32
          n_dense_rows::Int32

.. _details-structmi20__control:

mi20_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi20_control{T}
          f_arrays::Int32
          aggressive::Int32
          c_fail::Int32
          max_levels::Int32
          max_points::Int32
          reduction::T
          st_method::Int32
          st_parameter::T
          testing::Int32
          trunc_parameter::T
          coarse_solver::Int32
          coarse_solver_its::Int32
          damping::T
          err_tol::T
          levels::Int32
          pre_smoothing::Int32
          smoother::Int32
          post_smoothing::Int32
          v_iterations::Int32
          print_level::Int32
          print::Int32
          error::Int32
          one_pass_coarsen::Int32

.. _details-structmi20__solve__control:

mi20_solve__control structure
-----------------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi20_solve_control{T}
          abs_tol::T
          breakdown_tol::T
          gmres_restart::Int32
          init_guess::Bool
          krylov_solver::Int32
          max_its::Int32
          preconditioner_side::Int32
          rel_tol::T

.. _details-structmi20__info:

mi20_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi20_info{T}
          flag::Int32
          clevels::Int32
          cpoints::Int32
          cnnz::Int32
          stat::Int32
          getrf_info::Int32
          iterations::Int32
          residual::T

.. _details-structmi28__control:

mi28_control structure
----------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi28_control{T}
          f_arrays::Int32
          alpha::T
          check::Bool
          iorder::Int32
          iscale::Int32
          lowalpha::T
          maxshift::Int32
          rrt::Bool
          shift_factor::T
          shift_factor2::T
          small::T
          tau1::T
          tau2::T
          unit_error::Int32
          unit_warning::Int32

.. _details-structmi28__info:

mi28_info structure
-------------------

.. toctree::
	:hidden:

.. ref-code-block:: julia
	:class: doxyrest-overview-code-block

        struct mi28_info{T}
          band_after::Int32
          band_before::Int32
          dup::Int32
          flag::Int32
          flag61::Int32
          flag64::Int32
          flag68::Int32
          flag77::Int32
          nrestart::Int32
          nshift::Int32
          oor::Int32
          profile_before::T
          profile_after::T
          size_r::Clong
          stat::Int32
          alpha::T


detailed documentation
----------------------

HSL package control and info derived types as Julia structures.
See `HSL <https://www.hsl.rl.ac.uk/catalogue/>`_ 
documentation for further details.
