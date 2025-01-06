export gls_control_type

struct gls_control_type{T,INT}
  f_indexing::Bool
  lp::INT
  wp::INT
  mp::INT
  ldiag::INT
  btf::INT
  maxit::INT
  factor_blocking::INT
  solve_blas::INT
  la::INT
  la_int::INT
  maxla::INT
  pivoting::INT
  fill_in::INT
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

struct gls_ainfo_type{T,INT}
  flag::INT
  more::INT
  len_analyse::INT
  len_factorize::INT
  ncmpa::INT
  rank::INT
  drop::INT
  struc_rank::INT
  oor::INT
  dup::INT
  stat::INT
  lblock::INT
  sblock::INT
  tblock::INT
  ops::T
end

export gls_finfo_type

struct gls_finfo_type{T,INT}
  flag::INT
  more::INT
  size_factor::INT
  len_factorize::INT
  drop::INT
  rank::INT
  stat::INT
  ops::T
end

export gls_sinfo_type

struct gls_sinfo_type{INT}
  flag::INT
  more::INT
  stat::INT
end

export gls_initialize

function gls_initialize(::Type{Float32}, ::Type{Int32}, data, control)
  @ccall libgalahad_single.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{gls_control_type{Float32,Int32}})::Cvoid
end

function gls_initialize(::Type{Float32}, ::Type{Int64}, data, control)
  @ccall libgalahad_single_64.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gls_control_type{Float32,Int64}})::Cvoid
end

function gls_initialize(::Type{Float64}, ::Type{Int32}, data, control)
  @ccall libgalahad_double.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                          control::Ptr{gls_control_type{Float64,Int32}})::Cvoid
end

function gls_initialize(::Type{Float64}, ::Type{Int64}, data, control)
  @ccall libgalahad_double_64.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gls_control_type{Float64,Int64}})::Cvoid
end

function gls_initialize(::Type{Float128}, ::Type{Int32}, data, control)
  @ccall libgalahad_quadruple.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                             control::Ptr{gls_control_type{Float128,Int32}})::Cvoid
end

function gls_initialize(::Type{Float128}, ::Type{Int64}, data, control)
  @ccall libgalahad_quadruple_64.gls_initialize(data::Ptr{Ptr{Cvoid}},
                                                control::Ptr{gls_control_type{Float128,
                                                                              Int64}})::Cvoid
end

export gls_read_specfile

function gls_read_specfile(::Type{Float32}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_single.gls_read_specfile(control::Ptr{gls_control_type{Float32,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float32}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_single_64.gls_read_specfile(control::Ptr{gls_control_type{Float32,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float64}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_double.gls_read_specfile(control::Ptr{gls_control_type{Float64,Int32}},
                                             specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float64}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_double_64.gls_read_specfile(control::Ptr{gls_control_type{Float64,
                                                                              Int64}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float128}, ::Type{Int32}, control, specfile)
  @ccall libgalahad_quadruple.gls_read_specfile(control::Ptr{gls_control_type{Float128,
                                                                              Int32}},
                                                specfile::Ptr{Cchar})::Cvoid
end

function gls_read_specfile(::Type{Float128}, ::Type{Int64}, control, specfile)
  @ccall libgalahad_quadruple_64.gls_read_specfile(control::Ptr{gls_control_type{Float128,
                                                                                 Int64}},
                                                   specfile::Ptr{Cchar})::Cvoid
end

export gls_import

function gls_import(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.gls_import(control::Ptr{gls_control_type{Float32,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function gls_import(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.gls_import(control::Ptr{gls_control_type{Float32,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64})::Cvoid
end

function gls_import(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.gls_import(control::Ptr{gls_control_type{Float64,Int32}},
                                      data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function gls_import(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.gls_import(control::Ptr{gls_control_type{Float64,Int64}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int64})::Cvoid
end

function gls_import(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.gls_import(control::Ptr{gls_control_type{Float128,Int32}},
                                         data::Ptr{Ptr{Cvoid}}, status::Ptr{Int32})::Cvoid
end

function gls_import(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.gls_import(control::Ptr{gls_control_type{Float128,Int64}},
                                            data::Ptr{Ptr{Cvoid}},
                                            status::Ptr{Int64})::Cvoid
end

export gls_reset_control

function gls_reset_control(::Type{Float32}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_single.gls_reset_control(control::Ptr{gls_control_type{Float32,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function gls_reset_control(::Type{Float32}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_single_64.gls_reset_control(control::Ptr{gls_control_type{Float32,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function gls_reset_control(::Type{Float64}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_double.gls_reset_control(control::Ptr{gls_control_type{Float64,Int32}},
                                             data::Ptr{Ptr{Cvoid}},
                                             status::Ptr{Int32})::Cvoid
end

function gls_reset_control(::Type{Float64}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_double_64.gls_reset_control(control::Ptr{gls_control_type{Float64,
                                                                              Int64}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int64})::Cvoid
end

function gls_reset_control(::Type{Float128}, ::Type{Int32}, control, data, status)
  @ccall libgalahad_quadruple.gls_reset_control(control::Ptr{gls_control_type{Float128,
                                                                              Int32}},
                                                data::Ptr{Ptr{Cvoid}},
                                                status::Ptr{Int32})::Cvoid
end

function gls_reset_control(::Type{Float128}, ::Type{Int64}, control, data, status)
  @ccall libgalahad_quadruple_64.gls_reset_control(control::Ptr{gls_control_type{Float128,
                                                                                 Int64}},
                                                   data::Ptr{Ptr{Cvoid}},
                                                   status::Ptr{Int64})::Cvoid
end

export gls_information

function gls_information(::Type{Float32}, ::Type{Int32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single.gls_information(data::Ptr{Ptr{Cvoid}},
                                           ainfo::Ptr{gls_ainfo_type{Float32,Int32}},
                                           finfo::Ptr{gls_finfo_type{Float32,Int32}},
                                           sinfo::Ptr{gls_sinfo_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function gls_information(::Type{Float32}, ::Type{Int64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_single_64.gls_information(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ptr{gls_ainfo_type{Float32,Int64}},
                                              finfo::Ptr{gls_finfo_type{Float32,Int64}},
                                              sinfo::Ptr{gls_sinfo_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function gls_information(::Type{Float64}, ::Type{Int32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double.gls_information(data::Ptr{Ptr{Cvoid}},
                                           ainfo::Ptr{gls_ainfo_type{Float64,Int32}},
                                           finfo::Ptr{gls_finfo_type{Float64,Int32}},
                                           sinfo::Ptr{gls_sinfo_type{Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function gls_information(::Type{Float64}, ::Type{Int64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_double_64.gls_information(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ptr{gls_ainfo_type{Float64,Int64}},
                                              finfo::Ptr{gls_finfo_type{Float64,Int64}},
                                              sinfo::Ptr{gls_sinfo_type{Int64}},
                                              status::Ptr{Int64})::Cvoid
end

function gls_information(::Type{Float128}, ::Type{Int32}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_quadruple.gls_information(data::Ptr{Ptr{Cvoid}},
                                              ainfo::Ptr{gls_ainfo_type{Float128,Int32}},
                                              finfo::Ptr{gls_finfo_type{Float128,Int32}},
                                              sinfo::Ptr{gls_sinfo_type{Int32}},
                                              status::Ptr{Int32})::Cvoid
end

function gls_information(::Type{Float128}, ::Type{Int64}, data, ainfo, finfo, sinfo, status)
  @ccall libgalahad_quadruple_64.gls_information(data::Ptr{Ptr{Cvoid}},
                                                 ainfo::Ptr{gls_ainfo_type{Float128,Int64}},
                                                 finfo::Ptr{gls_finfo_type{Float128,Int64}},
                                                 sinfo::Ptr{gls_sinfo_type{Int64}},
                                                 status::Ptr{Int64})::Cvoid
end

export gls_finalize

function gls_finalize(::Type{Float32}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_single.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                        control::Ptr{gls_control_type{Float32,Int32}},
                                        status::Ptr{Int32})::Cvoid
end

function gls_finalize(::Type{Float32}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_single_64.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{gls_control_type{Float32,Int64}},
                                           status::Ptr{Int64})::Cvoid
end

function gls_finalize(::Type{Float64}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_double.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                        control::Ptr{gls_control_type{Float64,Int32}},
                                        status::Ptr{Int32})::Cvoid
end

function gls_finalize(::Type{Float64}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_double_64.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{gls_control_type{Float64,Int64}},
                                           status::Ptr{Int64})::Cvoid
end

function gls_finalize(::Type{Float128}, ::Type{Int32}, data, control, status)
  @ccall libgalahad_quadruple.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{gls_control_type{Float128,Int32}},
                                           status::Ptr{Int32})::Cvoid
end

function gls_finalize(::Type{Float128}, ::Type{Int64}, data, control, status)
  @ccall libgalahad_quadruple_64.gls_finalize(data::Ptr{Ptr{Cvoid}},
                                              control::Ptr{gls_control_type{Float128,Int64}},
                                              status::Ptr{Int64})::Cvoid
end
