export gls_control_type

mutable struct gls_control_type{T}
  f_indexing::Bool
  lp::Cint
  wp::Cint
  mp::Cint
  ldiag::Cint
  btf::Cint
  maxit::Cint
  factor_blocking::Cint
  solve_blas::Cint
  la::Cint
  la_int::Cint
  maxla::Cint
  pivoting::Cint
  fill_in::Cint
  multiplier::T
  reduce::T
  u::T
  switch_full::T
  drop::T
  tolerance::T
  cgce::T
  diagonal_pivoting::Bool
  struct_abort::Bool

  gls_control_type{T}() where T = new()
end

export gls_ainfo_type

mutable struct gls_ainfo_type{T}
  flag::Cint
  more::Cint
  len_analyse::Cint
  len_factorize::Cint
  ncmpa::Cint
  rank::Cint
  drop::Cint
  struc_rank::Cint
  oor::Cint
  dup::Cint
  stat::Cint
  lblock::Cint
  sblock::Cint
  tblock::Cint
  ops::T

  gls_ainfo_type{T}() where T = new()
end

export gls_finfo_type

mutable struct gls_finfo_type{T}
  flag::Cint
  more::Cint
  size_factor::Cint
  len_factorize::Cint
  drop::Cint
  rank::Cint
  stat::Cint
  ops::T

  gls_finfo_type{T}() where T = new()
end

export gls_sinfo_type

mutable struct gls_sinfo_type
  flag::Cint
  more::Cint
  stat::Cint

  gls_sinfo_type() = new()
end

export gls_initialize_s

function gls_initialize_s(data, control)
  @ccall libgalahad_single.gls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ref{gls_control_type{Float32}})::Cvoid
end

export gls_initialize

function gls_initialize(data, control)
  @ccall libgalahad_double.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ref{gls_control_type{Float64}})::Cvoid
end

export gls_read_specfile_s

function gls_read_specfile_s(control, specfile)
  @ccall libgalahad_single.gls_read_specfile_s(control::Ref{gls_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

export gls_read_specfile

function gls_read_specfile(control, specfile)
  @ccall libgalahad_double.gls_read_specfile(control::Ref{gls_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

export gls_import_s

function gls_import_s(control, data, status)
  @ccall libgalahad_single.gls_import_s(control::Ref{gls_control_type{Float32}}, data::Ptr{Ptr{Cvoid}},
                                        status::Ptr{Cint})::Cvoid
end

export gls_import

function gls_import(control, data, status)
  @ccall libgalahad_double.gls_import(control::Ref{gls_control_type{Float64}}, data::Ptr{Ptr{Cvoid}},
                                      status::Ptr{Cint})::Cvoid
end

export gls_reset_control_s

function gls_reset_control_s(control, data, status)
  @ccall libgalahad_single.gls_reset_control_s(control::Ref{gls_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

export gls_reset_control

function gls_reset_control(control, data, status)
  @ccall libgalahad_double.gls_reset_control(control::Ref{gls_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

export gls_information_s

function gls_information_s(data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single.gls_information_s(data::Ptr{Ptr{Cvoid}}, ainfo::Ref{gls_ainfo_type{Float32}},
                                             finfo::Ref{gls_finfo_type{Float32}}, sinfo::Ref{gls_sinfo_type},
                                             status::Ptr{Cint})::Cvoid
end

export gls_information

function gls_information(data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double.gls_information(data::Ptr{Ptr{Cvoid}}, ainfo::Ref{gls_ainfo_type{Float64}},
                                           finfo::Ref{gls_finfo_type{Float64}}, sinfo::Ref{gls_sinfo_type},
                                           status::Ptr{Cint})::Cvoid
end

export gls_finalize_s

function gls_finalize_s(data, control, status)
  @ccall libgalahad_single.gls_finalize_s(data::Ptr{Ptr{Cvoid}}, control::Ref{gls_control_type{Float32}},
                                          status::Ptr{Cint})::Cvoid
end

export gls_finalize

function gls_finalize(data, control, status)
  @ccall libgalahad_double.gls_finalize(data::Ptr{Ptr{Cvoid}}, control::Ref{gls_control_type{Float64}},
                                        status::Ptr{Cint})::Cvoid
end
