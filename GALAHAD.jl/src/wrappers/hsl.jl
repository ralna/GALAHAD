export ma48_control

mutable struct ma48_control{T}
  f_arrays::Cint
  multiplier::T
  u::T
  switch_::T
  drop::T
  tolerance::T
  cgce::T
  lp::Cint
  wp::Cint
  mp::Cint
  ldiag::Cint
  btf::Cint
  struct_::Cint
  maxit::Cint
  factor_blocking::Cint
  solve_blas::Cint
  pivoting::Cint
  diagonal_pivoting::Cint
  fill_in::Cint
  switch_mode::Cint

  ma48_control{T}() where T = new()
end

export ma48_ainfo

mutable struct ma48_ainfo{T}
  ops::T
  flag::Cint
  more::Cint
  lena_analyse::Clong
  lenj_analyse::Clong
  lena_factorize::Clong
  leni_factorize::Clong
  ncmpa::Cint
  rank::Cint
  drop::Clong
  struc_rank::Cint
  oor::Clong
  dup::Clong
  stat::Cint
  lblock::Cint
  sblock::Cint
  tblock::Clong

  ma48_ainfo{T}() where T = new()
end

export ma48_finfo

mutable struct ma48_finfo{T}
  ops::T
  flag::Cint
  more::Cint
  size_factor::Clong
  lena_factorize::Clong
  leni_factorize::Clong
  drop::Clong
  rank::Cint
  stat::Cint

  ma48_finfo{T}() where T = new()
end

export ma48_sinfo

mutable struct ma48_sinfo
  flag::Cint
  more::Cint
  stat::Cint

  ma48_sinfo() = new()
end

export ma57_control

mutable struct ma57_control{T}
  f_arrays::Cint
  multiplier::T
  reduce::T
  u::T
  static_tolerance::T
  static_level::T
  tolerance::T
  convergence::T
  consist::T
  lp::Cint
  wp::Cint
  mp::Cint
  sp::Cint
  ldiag::Cint
  nemin::Cint
  factorblocking::Cint
  solveblocking::Cint
  la::Cint
  liw::Cint
  maxla::Cint
  maxliw::Cint
  pivoting::Cint
  thresh::Cint
  ordering::Cint
  scaling::Cint
  rank_deficient::Cint
  ispare::NTuple{5,Cint}
  rspare::NTuple{10,T}

  ma57_control{T}() where T = new()
end

export ma57_ainfo

mutable struct ma57_ainfo{T}
  opsa::T
  opse::T
  flag::Cint
  more::Cint
  nsteps::Cint
  nrltot::Cint
  nirtot::Cint
  nrlnec::Cint
  nirnec::Cint
  nrladu::Cint
  niradu::Cint
  ncmpa::Cint
  ordering::Cint
  oor::Cint
  dup::Cint
  maxfrt::Cint
  stat::Cint
  ispare::NTuple{5,Cint}
  rspare::NTuple{10,T}

  ma57_ainfo{T}() where T = new()
end

export ma57_finfo

mutable struct ma57_finfo{T}
  opsa::T
  opse::T
  opsb::T
  maxchange::T
  smin::T
  smax::T
  flag::Cint
  more::Cint
  maxfrt::Cint
  nebdu::Cint
  nrlbdu::Cint
  nirbdu::Cint
  nrltot::Cint
  nirtot::Cint
  nrlnec::Cint
  nirnec::Cint
  ncmpbr::Cint
  ncmpbi::Cint
  ntwo::Cint
  neig::Cint
  delay::Cint
  signc::Cint
  static_::Cint
  modstep::Cint
  rank::Cint
  stat::Cint
  ispare::NTuple{5,Cint}
  rspare::NTuple{10,T}

  ma57_finfo{T}() where T = new()
end

export ma57_sinfo

mutable struct ma57_sinfo{T}
  cond::T
  cond2::T
  berr::T
  berr2::T
  error::T
  flag::Cint
  stat::Cint
  ispare::NTuple{5,Cint}
  rspare::NTuple{10,T}

  ma57_sinfo{T}() where T = new()
end

export ma77_control

mutable struct ma77_control{T}
  f_arrays::Cint
  print_level::Cint
  unit_diagnostics::Cint
  unit_error::Cint
  unit_warning::Cint
  bits::Cint
  buffer_lpage::NTuple{2,Cint}
  buffer_npage::NTuple{2,Cint}
  file_size::Clong
  maxstore::Clong
  storage::NTuple{3,Clong}
  nemin::Cint
  maxit::Cint
  infnorm::Cint
  thresh::T
  nb54::Cint
  action::Cint
  multiplier::T
  nb64::Cint
  nbi::Cint
  small::T
  static_::T
  storage_indef::Clong
  u::T
  umin::T
  consist_tol::T
  ispare::NTuple{5,Cint}
  lspare::NTuple{5,Clong}
  rspare::NTuple{5,T}

  ma77_control{T}() where T = new()
end

export ma77_info

mutable struct ma77_info{T}
  detlog::T
  detsign::Cint
  flag::Cint
  iostat::Cint
  matrix_dup::Cint
  matrix_rank::Cint
  matrix_outrange::Cint
  maxdepth::Cint
  maxfront::Cint
  minstore::Clong
  ndelay::Cint
  nfactor::Clong
  nflops::Clong
  niter::Cint
  nsup::Cint
  num_neg::Cint
  num_nothresh::Cint
  num_perturbed::Cint
  ntwo::Cint
  stat::Cint
  index::NTuple{4,Cint}
  nio_read::NTuple{2,Clong}
  nio_write::NTuple{2,Clong}
  nwd_read::NTuple{2,Clong}
  nwd_write::NTuple{2,Clong}
  num_file::NTuple{4,Cint}
  storage::NTuple{4,Clong}
  tree_nodes::Cint
  unit_restart::Cint
  unused::Cint
  usmall::T
  ispare::NTuple{5,Cint}
  lspare::NTuple{5,Clong}
  rspare::NTuple{5,T}

  ma77_info{T}() where T = new()
end

export ma86_control

mutable struct ma86_control{T}
  f_arrays::Cint
  diagnostics_level::Cint
  unit_diagnostics::Cint
  unit_error::Cint
  unit_warning::Cint
  nemin::Cint
  nb::Cint
  action::Cint
  nbi::Cint
  pool_size::Cint
  small_::T
  static_::T
  u::T
  umin::T
  scaling::Cint

  ma86_control{T}() where T = new()
end

export ma86_info

mutable struct ma86_info{T}
  detlog::T
  detsign::Cint
  flag::Cint
  matrix_rank::Cint
  maxdepth::Cint
  num_delay::Cint
  num_factor::Clong
  num_flops::Clong
  num_neg::Cint
  num_nodes::Cint
  num_nothresh::Cint
  num_perturbed::Cint
  num_two::Cint
  pool_size::Cint
  stat::Cint
  usmall::T

  ma86_info{T}() where T = new()
end

export ma87_control

mutable struct ma87_control{T}
  f_arrays::Cint
  diagnostics_level::Cint
  unit_diagnostics::Cint
  unit_error::Cint
  unit_warning::Cint
  nemin::Cint
  nb::Cint
  pool_size::Cint
  diag_zero_minus::T
  diag_zero_plus::T
  unused::NTuple{40,Cchar}

  ma87_control{T}() where T = new()
end

export ma87_info

mutable struct ma87_info{T}
  detlog::T
  flag::Cint
  maxdepth::Cint
  num_factor::Clong
  num_flops::Clong
  num_nodes::Cint
  pool_size::Cint
  stat::Cint
  num_zero::Cint
  unused::NTuple{40,Cchar}

  ma87_info{T}() where T = new()
end

export ma97_control

mutable struct ma97_control{T}
  f_arrays::Cint
  action::Cint
  nemin::Cint
  multiplier::T
  ordering::Cint
  print_level::Cint
  scaling::Cint
  small::T
  u::T
  unit_diagnostics::Cint
  unit_error::Cint
  unit_warning::Cint
  factor_min::Clong
  solve_blas3::Cint
  solve_min::Clong
  solve_mf::Cint
  consist_tol::T
  ispare::NTuple{5,Cint}
  rspare::NTuple{10,T}

  ma97_control{T}() where T = new()
end

export ma97_info

mutable struct ma97_info{T}
  flag::Cint
  flag68::Cint
  flag77::Cint
  matrix_dup::Cint
  matrix_rank::Cint
  matrix_outrange::Cint
  matrix_missing_diag::Cint
  maxdepth::Cint
  maxfront::Cint
  num_delay::Cint
  num_factor::Clong
  num_flops::Clong
  num_neg::Cint
  num_sup::Cint
  num_two::Cint
  ordering::Cint
  stat::Cint
  ispare::NTuple{5,Cint}
  rspare::NTuple{10,T}

  ma97_info{T}() where T = new()
end

export mc64_control

mutable struct mc64_control
  f_arrays::Cint
  lp::Cint
  wp::Cint
  sp::Cint
  ldiag::Cint
  checking::Cint

  mc64_control() = new()
end

export mc64_info

mutable struct mc64_info
  flag::Cint
  more::Cint
  strucrank::Cint
  stat::Cint

  mc64_info() = new()
end

export mc68_control

mutable struct mc68_control
  f_array_in::Cint
  f_array_out::Cint
  min_l_workspace::Cint
  lp::Cint
  wp::Cint
  mp::Cint
  nemin::Cint
  print_level::Cint
  row_full_thresh::Cint
  row_search::Cint

  mc68_control() = new()
end

export mc68_info

mutable struct mc68_info
  flag::Cint
  iostat::Cint
  stat::Cint
  out_range::Cint
  duplicate::Cint
  n_compressions::Cint
  n_zero_eigs::Cint
  l_workspace::Clong
  zb01_info::Cint
  n_dense_rows::Cint

  mc68_info() = new()
end

export mi20_control

mutable struct mi20_control{T}
  f_arrays::Cint
  aggressive::Cint
  c_fail::Cint
  max_levels::Cint
  max_points::Cint
  reduction::T
  st_method::Cint
  st_parameter::T
  testing::Cint
  trunc_parameter::T
  coarse_solver::Cint
  coarse_solver_its::Cint
  damping::T
  err_tol::T
  levels::Cint
  pre_smoothing::Cint
  smoother::Cint
  post_smoothing::Cint
  v_iterations::Cint
  print_level::Cint
  print::Cint
  error::Cint
  one_pass_coarsen::Cint

  mi20_control{T}() where T = new()
end

export mi20_solve_control

mutable struct mi20_solve_control{T}
  abs_tol::T
  breakdown_tol::T
  gmres_restart::Cint
  init_guess::Bool
  krylov_solver::Cint
  max_its::Cint
  preconditioner_side::Cint
  rel_tol::T

  mi20_solve_control{T}() where T = new()
end

export mi20_info

mutable struct mi20_info{T}
  flag::Cint
  clevels::Cint
  cpoints::Cint
  cnnz::Cint
  stat::Cint
  getrf_info::Cint
  iterations::Cint
  residual::T

  mi20_info{T}() where T = new()
end

export mi28_control

mutable struct mi28_control{T}
  f_arrays::Cint
  alpha::T
  check::Bool
  iorder::Cint
  iscale::Cint
  lowalpha::T
  maxshift::Cint
  rrt::Bool
  shift_factor::T
  shift_factor2::T
  small::T
  tau1::T
  tau2::T
  unit_error::Cint
  unit_warning::Cint

  mi28_control{T}() where T = new()
end

export mi28_info

mutable struct mi28_info{T}
  band_after::Cint
  band_before::Cint
  dup::Cint
  flag::Cint
  flag61::Cint
  flag64::Cint
  flag68::Cint
  flag77::Cint
  nrestart::Cint
  nshift::Cint
  oor::Cint
  profile_before::T
  profile_after::T
  size_r::Clong
  stat::Cint
  alpha::T

  mi28_info{T}() where T = new()
end
