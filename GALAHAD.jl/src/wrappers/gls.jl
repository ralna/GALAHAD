export gls_control_type

struct gls_control_type{T}
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
end

export gls_ainfo_type

struct gls_ainfo_type{T}
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
end

export gls_finfo_type

struct gls_finfo_type{T}
  flag::Cint
  more::Cint
  size_factor::Cint
  len_factorize::Cint
  drop::Cint
  rank::Cint
  stat::Cint
  ops::T
end

export gls_sinfo_type

struct gls_sinfo_type
  flag::Cint
  more::Cint
  stat::Cint
end

export gls_initialize

function gls_initialize(::Type{Float32}, data, control)
  @ccall libgalahad_single.gls_initialize_s(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{gls_control_type{Float32}})::Cvoid
end

function gls_initialize(::Type{Float64}, data, control)
  @ccall libgalahad_double.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{gls_control_type{Float64}})::Cvoid
end

function gls_initialize(::Type{Float128}, data, control)
  @ccall libgalahad_quadruple.gls_initialize_q(data::Ptr{Ptr{Cvoid}},
                                               control::Ptr{gls_control_type{Float128}})::Cvoid
end

export gls_read_specfile

function gls_read_specfile(::Type{Float32}, control, specfile)
  @ccall libgalahad_single.gls_read_specfile_s(control::Ptr{gls_control_type{Float32}},
                                               specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float64}, control, specfile)
  @ccall libgalahad_double.gls_read_specfile(control::Ptr{gls_control_type{Float64}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float128}, control, specfile)
  @ccall libgalahad_quadruple.gls_read_specfile_q(control::Ptr{gls_control_type{Float128}},
                                                  specfile::Ptr{Cchar})::Cvoid
end

export gls_import

function gls_import(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.gls_import_s(control::Ptr{gls_control_type{Float32}},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

function gls_import(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.gls_import(control::Ptr{gls_control_type{Float64}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

function gls_import(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.gls_import_q(control::Ptr{gls_control_type{Float128}},
                                           data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint})::Cvoid
end

export gls_reset_control

function gls_reset_control(::Type{Float32}, control, data, status)
  @ccall libgalahad_single.gls_reset_control_s(control::Ptr{gls_control_type{Float32}},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function gls_reset_control(::Type{Float64}, control, data, status)
  @ccall libgalahad_double.gls_reset_control(control::Ptr{gls_control_type{Float64}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Cint})::Cvoid
end

function gls_reset_control(::Type{Float128}, control, data, status)
  @ccall libgalahad_quadruple.gls_reset_control_q(control::Ptr{gls_control_type{Float128}},
                                                  data::Ptr{Ptr{Cvoid}},
                                                  status::Ptr{Cint})::Cvoid
end

export gls_information

function gls_information(::Type{Float32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single.gls_information_s(data::Ptr{Ptr{Cvoid}},
                                             ainfo::Ptr{gls_ainfo_type{Float32}},
                                             finfo::Ptr{gls_finfo_type{Float32}},
                                             sinfo::Ptr{gls_sinfo_type},
                                             status::Ptr{Cint})::Cvoid
end

function gls_information(::Type{Float64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double.gls_information(data::Ptr{Ptr{Cvoid}},
                                           ainfo::Ptr{gls_ainfo_type{Float64}},
                                           finfo::Ptr{gls_finfo_type{Float64}},
                                           sinfo::Ptr{gls_sinfo_type},
                                           status::Ptr{Cint})::Cvoid
end

function gls_information(::Type{Float128}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_quadruple.gls_information_q(data::Ptr{Ptr{Cvoid}},
                                                ainfo::Ptr{gls_ainfo_type{Float128}},
                                                finfo::Ptr{gls_finfo_type{Float128}},
                                                sinfo::Ptr{gls_sinfo_type},
                                                status::Ptr{Cint})::Cvoid
end

export gls_finalize

function gls_finalize(::Type{Float32}, data, control, status)
  @ccall libgalahad_single.gls_finalize_s(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{gls_control_type{Float32}},
                                          status::Ptr{Cint})::Cvoid
end

function gls_finalize(::Type{Float64}, data, control, status)
  @ccall libgalahad_double.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                        control::Ptr{gls_control_type{Float64}},
                                        status::Ptr{Cint})::Cvoid
end

function gls_finalize(::Type{Float128}, data, control, status)
  @ccall libgalahad_quadruple.gls_finalize_q(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gls_control_type{Float128}},
                                             status::Ptr{Cint})::Cvoid
end
