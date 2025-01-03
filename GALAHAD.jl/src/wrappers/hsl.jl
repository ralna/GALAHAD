
export ma48_control

struct ma48_control{T,INT}
  f_arrays::INT
  multiplier::T
  u::T
  switch_::T
  drop::T
  tolerance::T
  cgce::T
  lp::INT
  wp::INT
  mp::INT
  ldiag::INT
  btf::INT
  struct_::INT
  maxit::INT
  factor_blocking::INT
  solve_blas::INT
  pivoting::INT
  diagonal_pivoting::INT
  fill_in::INT
  switch_mode::INT
end

export ma48_ainfo

struct ma48_ainfo{T,INT}
  ops::T
  flag::INT
  more::INT
  lena_analyse::Int64
  lenj_analyse::Int64
  lena_factorize::Int64
  leni_factorize::Int64
  ncmpa::INT
  rank::INT
  drop::Int64
  struc_rank::INT
  oor::Int64
  dup::Int64
  stat::INT
  lblock::INT
  sblock::INT
  tblock::Int64
end

export ma48_finfo

struct ma48_finfo{T,INT}
  ops::T
  flag::INT
  more::INT
  size_factor::Int64
  lena_factorize::Int64
  leni_factorize::Int64
  drop::Int64
  rank::INT
  stat::INT
end

export ma48_sinfo

struct ma48_sinfo{INT}
  flag::INT
  more::INT
  stat::INT
end

export ma57_control

struct ma57_control{T,INT}
  f_arrays::INT
  multiplier::T
  reduce::T
  u::T
  static_tolerance::T
  static_level::T
  tolerance::T
  convergence::T
  consist::T
  lp::INT
  wp::INT
  mp::INT
  sp::INT
  ldiag::INT
  nemin::INT
  factorblocking::INT
  solveblocking::INT
  la::INT
  liw::INT
  maxla::INT
  maxliw::INT
  pivoting::INT
  thresh::INT
  ordering::INT
  scaling::INT
  rank_deficient::INT
  ispare::NTuple{5,INT}
  rspare::NTuple{10,T}
end

export ma57_ainfo

struct ma57_ainfo{T,INT}
  opsa::T
  opse::T
  flag::INT
  more::INT
  nsteps::INT
  nrltot::INT
  nirtot::INT
  nrlnec::INT
  nirnec::INT
  nrladu::INT
  niradu::INT
  ncmpa::INT
  ordering::INT
  oor::INT
  dup::INT
  maxfrt::INT
  stat::INT
  ispare::NTuple{5,INT}
  rspare::NTuple{10,T}
end

export ma57_finfo

struct ma57_finfo{T,INT}
  opsa::T
  opse::T
  opsb::T
  maxchange::T
  smin::T
  smax::T
  flag::INT
  more::INT
  maxfrt::INT
  nebdu::INT
  nrlbdu::INT
  nirbdu::INT
  nrltot::INT
  nirtot::INT
  nrlnec::INT
  nirnec::INT
  ncmpbr::INT
  ncmpbi::INT
  ntwo::INT
  neig::INT
  delay::INT
  signc::INT
  static_::INT
  modstep::INT
  rank::INT
  stat::INT
  ispare::NTuple{5,INT}
  rspare::NTuple{10,T}
end

export ma57_sinfo

struct ma57_sinfo{T,INT}
  cond::T
  cond2::T
  berr::T
  berr2::T
  error::T
  flag::INT
  stat::INT
  ispare::NTuple{5,INT}
  rspare::NTuple{10,T}
end

export ma77_control

struct ma77_control{T,INT}
  f_arrays::INT
  print_level::INT
  unit_diagnostics::INT
  unit_error::INT
  unit_warning::INT
  bits::INT
  buffer_lpage::NTuple{2,INT}
  buffer_npage::NTuple{2,INT}
  file_size::Int64
  maxstore::Int64
  storage::NTuple{3,Int64}
  nemin::INT
  maxit::INT
  infnorm::INT
  thresh::T
  nb54::INT
  action::INT
  multiplier::T
  nb64::INT
  nbi::INT
  small::T
  static_::T
  storage_indef::Int64
  u::T
  umin::T
  consist_tol::T
  ispare::NTuple{5,INT}
  lspare::NTuple{5,Int64}
  rspare::NTuple{5,T}
end

export ma77_info

struct ma77_info{T,INT}
  detlog::T
  detsign::INT
  flag::INT
  iostat::INT
  matrix_dup::INT
  matrix_rank::INT
  matrix_outrange::INT
  maxdepth::INT
  maxfront::INT
  minstore::Int64
  ndelay::INT
  nfactor::Int64
  nflops::Int64
  niter::INT
  nsup::INT
  num_neg::INT
  num_nothresh::INT
  num_perturbed::INT
  ntwo::INT
  stat::INT
  index::NTuple{4,INT}
  nio_read::NTuple{2,Int64}
  nio_write::NTuple{2,Int64}
  nwd_read::NTuple{2,Int64}
  nwd_write::NTuple{2,Int64}
  num_file::NTuple{4,INT}
  storage::NTuple{4,Int64}
  tree_nodes::INT
  unit_restart::INT
  unused::INT
  usmall::T
  ispare::NTuple{5,INT}
  lspare::NTuple{5,Int64}
  rspare::NTuple{5,T}
end

export ma86_control

struct ma86_control{T,INT}
  f_arrays::INT
  diagnostics_level::INT
  unit_diagnostics::INT
  unit_error::INT
  unit_warning::INT
  nemin::INT
  nb::INT
  action::INT
  nbi::INT
  pool_size::INT
  small_::T
  static_::T
  u::T
  umin::T
  scaling::INT
end

export ma86_info

struct ma86_info{T,INT}
  detlog::T
  detsign::INT
  flag::INT
  matrix_rank::INT
  maxdepth::INT
  num_delay::INT
  num_factor::Int64
  num_flops::Int64
  num_neg::INT
  num_nodes::INT
  num_nothresh::INT
  num_perturbed::INT
  num_two::INT
  pool_size::INT
  stat::INT
  usmall::T
end

export ma87_control

struct ma87_control{T,INT}
  f_arrays::INT
  diagnostics_level::INT
  unit_diagnostics::INT
  unit_error::INT
  unit_warning::INT
  nemin::INT
  nb::INT
  pool_size::INT
  diag_zero_minus::T
  diag_zero_plus::T
  unused::NTuple{40,Cchar}
end

export ma87_info

struct ma87_info{T,INT}
  detlog::T
  flag::INT
  maxdepth::INT
  num_factor::Int64
  num_flops::Int64
  num_nodes::INT
  pool_size::INT
  stat::INT
  num_zero::INT
  unused::NTuple{40,Cchar}
end

export ma97_control

struct ma97_control{T,INT}
  f_arrays::INT
  action::INT
  nemin::INT
  multiplier::T
  ordering::INT
  print_level::INT
  scaling::INT
  small::T
  u::T
  unit_diagnostics::INT
  unit_error::INT
  unit_warning::INT
  factor_min::Int64
  solve_blas3::INT
  solve_min::Int64
  solve_mf::INT
  consist_tol::T
  ispare::NTuple{5,INT}
  rspare::NTuple{10,T}
end

export ma97_info

struct ma97_info{T,INT}
  flag::INT
  flag68::INT
  flag77::INT
  matrix_dup::INT
  matrix_rank::INT
  matrix_outrange::INT
  matrix_missing_diag::INT
  maxdepth::INT
  maxfront::INT
  num_delay::INT
  num_factor::Int64
  num_flops::Int64
  num_neg::INT
  num_sup::INT
  num_two::INT
  ordering::INT
  stat::INT
  maxsupernode::INT
  ispare::NTuple{4,INT}
  rspare::NTuple{10,T}
end

export mc64_control

struct mc64_control{INT}
  f_arrays::INT
  lp::INT
  wp::INT
  sp::INT
  ldiag::INT
  checking::INT
end

export mc64_info

struct mc64_info{INT}
  flag::INT
  more::INT
  strucrank::INT
  stat::INT
end

export mc68_control

struct mc68_control{INT}
  f_array_in::INT
  f_array_out::INT
  min_l_workspace::INT
  lp::INT
  wp::INT
  mp::INT
  nemin::INT
  print_level::INT
  row_full_thresh::INT
  row_search::INT
end

export mc68_info

struct mc68_info{INT}
  flag::INT
  iostat::INT
  stat::INT
  out_range::INT
  duplicate::INT
  n_compressions::INT
  n_zero_eigs::INT
  l_workspace::Int64
  zb01_info::INT
  n_dense_rows::INT
end

export mi20_control

struct mi20_control{T,INT}
  f_arrays::INT
  aggressive::INT
  c_fail::INT
  max_levels::INT
  max_points::INT
  reduction::T
  st_method::INT
  st_parameter::T
  testing::INT
  trunc_parameter::T
  coarse_solver::INT
  coarse_solver_its::INT
  damping::T
  err_tol::T
  levels::INT
  pre_smoothing::INT
  smoother::INT
  post_smoothing::INT
  v_iterations::INT
  print_level::INT
  print::INT
  error::INT
  one_pass_coarsen::INT
end

export mi20_solve_control

struct mi20_solve_control{T,INT}
  abs_tol::T
  breakdown_tol::T
  gmres_restart::INT
  init_guess::Bool
  krylov_solver::INT
  max_its::INT
  preconditioner_side::INT
  rel_tol::T
end

export mi20_info

struct mi20_info{T,INT}
  flag::INT
  clevels::INT
  cpoints::INT
  cnnz::INT
  stat::INT
  getrf_info::INT
  iterations::INT
  residual::T
end

export mi28_control

struct mi28_control{T,INT}
  f_arrays::INT
  alpha::T
  check::Bool
  iorder::INT
  iscale::INT
  lowalpha::T
  maxshift::INT
  rrt::Bool
  shift_factor::T
  shift_factor2::T
  small::T
  tau1::T
  tau2::T
  unit_error::INT
  unit_warning::INT
end

export mi28_info

struct mi28_info{T,INT}
  band_after::INT
  band_before::INT
  dup::INT
  flag::INT
  flag61::INT
  flag64::INT
  flag68::INT
  flag77::INT
  nrestart::INT
  nshift::INT
  oor::INT
  profile_before::T
  profile_after::T
  size_r::Int64
  stat::INT
  alpha::T
end
