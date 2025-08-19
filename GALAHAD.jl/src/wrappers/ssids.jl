export ssids_control_type

struct ssids_control_type{T,INT}
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
  max_load_inbalance::Float32
  gpu_perf_coeff::Float32
  scaling::INT
  small_subtree_threshold::Int64
  cpu_block_size::INT
  action::Bool
  pivot_method::INT
  small::T
  u::T
  nodend_control::nodend_control_type{INT}
  nstream::INT
  multiplier::T
  min_loadbalance::Float32
  failed_pivot_method::INT
end

export ssids_inform_type

struct ssids_inform_type{T,INT}
  flag::INT
  matrix_dup::INT
  matrix_missing_diag::INT
  matrix_outrange::INT
  matrix_rank::INT
  maxdepth::INT
  maxfront::INT
  maxsupernode::INT
  num_delay::INT
  num_factor::Int64
  num_flops::Int64
  num_neg::INT
  num_sup::INT
  num_two::INT
  stat::INT
  cuda_error::INT
  cublas_error::INT
  nodend_inform::nodend_inform_type{T,INT}
  not_first_pass::INT
  not_second_pass::INT
  nparts::INT
  cpu_flops::Int64
  gpu_flops::Int64
end

export spral_ssids_default_control

function spral_ssids_default_control(::Type{Float32}, ::Type{Int32}, control)
  @ccall libgalahad_single.spral_ssids_default_control_s(control::Ptr{ssids_control_type{Float32,
                                                                                          Int32}})::Cvoid
end

function spral_ssids_default_control(::Type{Float32}, ::Type{Int64}, control)
  @ccall libgalahad_single_64.spral_ssids_default_control_s_64(control::Ptr{ssids_control_type{Float32,
                                                                                                Int64}})::Cvoid
end

function spral_ssids_default_control(::Type{Float64}, ::Type{Int32}, control)
  @ccall libgalahad_double.spral_ssids_default_control(control::Ptr{ssids_control_type{Float64,
                                                                                        Int32}})::Cvoid
end

function spral_ssids_default_control(::Type{Float64}, ::Type{Int64}, control)
  @ccall libgalahad_double_64.spral_ssids_default_control_64(control::Ptr{ssids_control_type{Float64,
                                                                                              Int64}})::Cvoid
end

function spral_ssids_default_control(::Type{Float128}, ::Type{Int32}, control)
  @ccall libgalahad_quadruple.spral_ssids_default_control_q(control::Ptr{ssids_control_type{Float128,
                                                                                             Int32}})::Cvoid
end

function spral_ssids_default_control(::Type{Float128}, ::Type{Int64}, control)
  @ccall libgalahad_quadruple_64.spral_ssids_default_control_q_64(control::Ptr{ssids_control_type{Float128,
                                                                                                   Int64}})::Cvoid
end

export spral_ssids_analyse

function spral_ssids_analyse(::Type{Float32}, ::Type{Int32}, check, n, order, ptr, row, val,
                             akeep, control, inform)
  @ccall libgalahad_single.spral_ssids_analyse_s(check::Bool, n::Int32, order::Ptr{Int32},
                                                 ptr::Ptr{Int64}, row::Ptr{Int32},
                                                 val::Ptr{Float32}, akeep::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{ssids_control_type{Float32,
                                                                                  Int32}},
                                                 inform::Ptr{ssids_inform_type{Float32,
                                                                                Int32}})::Cvoid
end

function spral_ssids_analyse(::Type{Float32}, ::Type{Int64}, check, n, order, ptr, row, val,
                             akeep, control, inform)
  @ccall libgalahad_single_64.spral_ssids_analyse_s_64(check::Bool, n::Int64,
                                                       order::Ptr{Int64}, ptr::Ptr{Int64},
                                                       row::Ptr{Int64}, val::Ptr{Float32},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       control::Ptr{ssids_control_type{Float32,
                                                                                        Int64}},
                                                       inform::Ptr{ssids_inform_type{Float32,
                                                                                      Int64}})::Cvoid
end

function spral_ssids_analyse(::Type{Float64}, ::Type{Int32}, check, n, order, ptr, row, val,
                             akeep, control, inform)
  @ccall libgalahad_double.spral_ssids_analyse(check::Bool, n::Int32, order::Ptr{Int32},
                                               ptr::Ptr{Int64}, row::Ptr{Int32},
                                               val::Ptr{Float64}, akeep::Ptr{Ptr{Cvoid}},
                                               control::Ptr{ssids_control_type{Float64,
                                                                                Int32}},
                                               inform::Ptr{ssids_inform_type{Float64,
                                                                              Int32}})::Cvoid
end

function spral_ssids_analyse(::Type{Float64}, ::Type{Int64}, check, n, order, ptr, row, val,
                             akeep, control, inform)
  @ccall libgalahad_double_64.spral_ssids_analyse_64(check::Bool, n::Int64,
                                                     order::Ptr{Int64}, ptr::Ptr{Int64},
                                                     row::Ptr{Int64}, val::Ptr{Float64},
                                                     akeep::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{ssids_control_type{Float64,
                                                                                      Int64}},
                                                     inform::Ptr{ssids_inform_type{Float64,
                                                                                    Int64}})::Cvoid
end

function spral_ssids_analyse(::Type{Float128}, ::Type{Int32}, check, n, order, ptr, row,
                             val, akeep, control, inform)
  @ccall libgalahad_quadruple.spral_ssids_analyse_q(check::Bool, n::Int32,
                                                    order::Ptr{Int32}, ptr::Ptr{Int64},
                                                    row::Ptr{Int32}, val::Ptr{Float128},
                                                    akeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{ssids_control_type{Float128,
                                                                                     Int32}},
                                                    inform::Ptr{ssids_inform_type{Float128,
                                                                                   Int32}})::Cvoid
end

function spral_ssids_analyse(::Type{Float128}, ::Type{Int64}, check, n, order, ptr, row,
                             val, akeep, control, inform)
  @ccall libgalahad_quadruple_64.spral_ssids_analyse_q_64(check::Bool, n::Int64,
                                                          order::Ptr{Int64},
                                                          ptr::Ptr{Int64}, row::Ptr{Int64},
                                                          val::Ptr{Float128},
                                                          akeep::Ptr{Ptr{Cvoid}},
                                                          control::Ptr{ssids_control_type{Float128,
                                                                                           Int64}},
                                                          inform::Ptr{ssids_inform_type{Float128,
                                                                                         Int64}})::Cvoid
end

export spral_ssids_analyse_ptr32

function spral_ssids_analyse_ptr32(::Type{Float32}, ::Type{Int32}, check, n, order, ptr,
                                   row, val, akeep, control, inform)
  @ccall libgalahad_single.spral_ssids_analyse_ptr32_s(check::Bool, n::Int32,
                                                       order::Ptr{Int32}, ptr::Ptr{Int32},
                                                       row::Ptr{Int32}, val::Ptr{Float32},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       control::Ptr{ssids_control_type{Float32,
                                                                                        Int32}},
                                                       inform::Ptr{ssids_inform_type{Float32,
                                                                                      Int32}})::Cvoid
end

function spral_ssids_analyse_ptr32(::Type{Float32}, ::Type{Int64}, check, n, order, ptr,
                                   row, val, akeep, control, inform)
  @ccall libgalahad_single_64.spral_ssids_analyse_ptr32_s_64(check::Bool, n::Int64,
                                                             order::Ptr{Int64},
                                                             ptr::Ptr{Int64},
                                                             row::Ptr{Int64},
                                                             val::Ptr{Float32},
                                                             akeep::Ptr{Ptr{Cvoid}},
                                                             control::Ptr{ssids_control_type{Float32,
                                                                                              Int64}},
                                                             inform::Ptr{ssids_inform_type{Float32,
                                                                                            Int64}})::Cvoid
end

function spral_ssids_analyse_ptr32(::Type{Float64}, ::Type{Int32}, check, n, order, ptr,
                                   row, val, akeep, control, inform)
  @ccall libgalahad_double.spral_ssids_analyse_ptr32(check::Bool, n::Int32,
                                                     order::Ptr{Int32}, ptr::Ptr{Int32},
                                                     row::Ptr{Int32}, val::Ptr{Float64},
                                                     akeep::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{ssids_control_type{Float64,
                                                                                      Int32}},
                                                     inform::Ptr{ssids_inform_type{Float64,
                                                                                    Int32}})::Cvoid
end

function spral_ssids_analyse_ptr32(::Type{Float64}, ::Type{Int64}, check, n, order, ptr,
                                   row, val, akeep, control, inform)
  @ccall libgalahad_double_64.spral_ssids_analyse_ptr32_64(check::Bool, n::Int64,
                                                           order::Ptr{Int64},
                                                           ptr::Ptr{Int64}, row::Ptr{Int64},
                                                           val::Ptr{Float64},
                                                           akeep::Ptr{Ptr{Cvoid}},
                                                           control::Ptr{ssids_control_type{Float64,
                                                                                            Int64}},
                                                           inform::Ptr{ssids_inform_type{Float64,
                                                                                          Int64}})::Cvoid
end

function spral_ssids_analyse_ptr32(::Type{Float128}, ::Type{Int32}, check, n, order, ptr,
                                   row, val, akeep, control, inform)
  @ccall libgalahad_quadruple.spral_ssids_analyse_ptr32_q(check::Bool, n::Int32,
                                                          order::Ptr{Int32},
                                                          ptr::Ptr{Int32}, row::Ptr{Int32},
                                                          val::Ptr{Float128},
                                                          akeep::Ptr{Ptr{Cvoid}},
                                                          control::Ptr{ssids_control_type{Float128,
                                                                                           Int32}},
                                                          inform::Ptr{ssids_inform_type{Float128,
                                                                                         Int32}})::Cvoid
end

function spral_ssids_analyse_ptr32(::Type{Float128}, ::Type{Int64}, check, n, order, ptr,
                                   row, val, akeep, control, inform)
  @ccall libgalahad_quadruple_64.spral_ssids_analyse_ptr32_q_64(check::Bool, n::Int64,
                                                                order::Ptr{Int64},
                                                                ptr::Ptr{Int64},
                                                                row::Ptr{Int64},
                                                                val::Ptr{Float128},
                                                                akeep::Ptr{Ptr{Cvoid}},
                                                                control::Ptr{ssids_control_type{Float128,
                                                                                                 Int64}},
                                                                inform::Ptr{ssids_inform_type{Float128,
                                                                                               Int64}})::Cvoid
end

export spral_ssids_analyse_coord

function spral_ssids_analyse_coord(::Type{Float32}, ::Type{Int32}, n, order, ne, row, col,
                                   val, akeep, control, inform)
  @ccall libgalahad_single.spral_ssids_analyse_coord_s(n::Int32, order::Ptr{Int32},
                                                       ne::Int64, row::Ptr{Int32},
                                                       col::Ptr{Int32}, val::Ptr{Float32},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       control::Ptr{ssids_control_type{Float32,
                                                                                        Int32}},
                                                       inform::Ptr{ssids_inform_type{Float32,
                                                                                      Int32}})::Cvoid
end

function spral_ssids_analyse_coord(::Type{Float32}, ::Type{Int64}, n, order, ne, row, col,
                                   val, akeep, control, inform)
  @ccall libgalahad_single_64.spral_ssids_analyse_coord_s_64(n::Int64, order::Ptr{Int64},
                                                             ne::Int64, row::Ptr{Int64},
                                                             col::Ptr{Int64},
                                                             val::Ptr{Float32},
                                                             akeep::Ptr{Ptr{Cvoid}},
                                                             control::Ptr{ssids_control_type{Float32,
                                                                                              Int64}},
                                                             inform::Ptr{ssids_inform_type{Float32,
                                                                                            Int64}})::Cvoid
end

function spral_ssids_analyse_coord(::Type{Float64}, ::Type{Int32}, n, order, ne, row, col,
                                   val, akeep, control, inform)
  @ccall libgalahad_double.spral_ssids_analyse_coord(n::Int32, order::Ptr{Int32}, ne::Int64,
                                                     row::Ptr{Int32}, col::Ptr{Int32},
                                                     val::Ptr{Float64},
                                                     akeep::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{ssids_control_type{Float64,
                                                                                      Int32}},
                                                     inform::Ptr{ssids_inform_type{Float64,
                                                                                    Int32}})::Cvoid
end

function spral_ssids_analyse_coord(::Type{Float64}, ::Type{Int64}, n, order, ne, row, col,
                                   val, akeep, control, inform)
  @ccall libgalahad_double_64.spral_ssids_analyse_coord_64(n::Int64, order::Ptr{Int64},
                                                           ne::Int64, row::Ptr{Int64},
                                                           col::Ptr{Int64},
                                                           val::Ptr{Float64},
                                                           akeep::Ptr{Ptr{Cvoid}},
                                                           control::Ptr{ssids_control_type{Float64,
                                                                                            Int64}},
                                                           inform::Ptr{ssids_inform_type{Float64,
                                                                                          Int64}})::Cvoid
end

function spral_ssids_analyse_coord(::Type{Float128}, ::Type{Int32}, n, order, ne, row, col,
                                   val, akeep, control, inform)
  @ccall libgalahad_quadruple.spral_ssids_analyse_coord_q(n::Int32, order::Ptr{Int32},
                                                          ne::Int64, row::Ptr{Int32},
                                                          col::Ptr{Int32},
                                                          val::Ptr{Float128},
                                                          akeep::Ptr{Ptr{Cvoid}},
                                                          control::Ptr{ssids_control_type{Float128,
                                                                                           Int32}},
                                                          inform::Ptr{ssids_inform_type{Float128,
                                                                                         Int32}})::Cvoid
end

function spral_ssids_analyse_coord(::Type{Float128}, ::Type{Int64}, n, order, ne, row, col,
                                   val, akeep, control, inform)
  @ccall libgalahad_quadruple_64.spral_ssids_analyse_coord_q_64(n::Int64, order::Ptr{Int64},
                                                                ne::Int64, row::Ptr{Int64},
                                                                col::Ptr{Int64},
                                                                val::Ptr{Float128},
                                                                akeep::Ptr{Ptr{Cvoid}},
                                                                control::Ptr{ssids_control_type{Float128,
                                                                                                 Int64}},
                                                                inform::Ptr{ssids_inform_type{Float128,
                                                                                               Int64}})::Cvoid
end

export spral_ssids_factor

function spral_ssids_factor(::Type{Float32}, ::Type{Int32}, posdef, ptr, row, val, scale,
                            akeep, fkeep, control, inform)
  @ccall libgalahad_single.spral_ssids_factor_s(posdef::Bool, ptr::Ptr{Int64},
                                                row::Ptr{Int32}, val::Ptr{Float32},
                                                scale::Ptr{Float32}, akeep::Ptr{Cvoid},
                                                fkeep::Ptr{Ptr{Cvoid}},
                                                control::Ptr{ssids_control_type{Float32,
                                                                                 Int32}},
                                                inform::Ptr{ssids_inform_type{Float32,
                                                                               Int32}})::Cvoid
end

function spral_ssids_factor(::Type{Float32}, ::Type{Int64}, posdef, ptr, row, val, scale,
                            akeep, fkeep, control, inform)
  @ccall libgalahad_single_64.spral_ssids_factor_s_64(posdef::Bool, ptr::Ptr{Int64},
                                                      row::Ptr{Int64}, val::Ptr{Float32},
                                                      scale::Ptr{Float32},
                                                      akeep::Ptr{Cvoid},
                                                      fkeep::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{ssids_control_type{Float32,
                                                                                       Int64}},
                                                      inform::Ptr{ssids_inform_type{Float32,
                                                                                     Int64}})::Cvoid
end

function spral_ssids_factor(::Type{Float64}, ::Type{Int32}, posdef, ptr, row, val, scale,
                            akeep, fkeep, control, inform)
  @ccall libgalahad_double.spral_ssids_factor(posdef::Bool, ptr::Ptr{Int64},
                                              row::Ptr{Int32}, val::Ptr{Float64},
                                              scale::Ptr{Float64}, akeep::Ptr{Cvoid},
                                              fkeep::Ptr{Ptr{Cvoid}},
                                              control::Ptr{ssids_control_type{Float64,
                                                                               Int32}},
                                              inform::Ptr{ssids_inform_type{Float64,Int32}})::Cvoid
end

function spral_ssids_factor(::Type{Float64}, ::Type{Int64}, posdef, ptr, row, val, scale,
                            akeep, fkeep, control, inform)
  @ccall libgalahad_double_64.spral_ssids_factor_64(posdef::Bool, ptr::Ptr{Int64},
                                                    row::Ptr{Int64}, val::Ptr{Float64},
                                                    scale::Ptr{Float64}, akeep::Ptr{Cvoid},
                                                    fkeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{ssids_control_type{Float64,
                                                                                     Int64}},
                                                    inform::Ptr{ssids_inform_type{Float64,
                                                                                   Int64}})::Cvoid
end

function spral_ssids_factor(::Type{Float128}, ::Type{Int32}, posdef, ptr, row, val, scale,
                            akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple.spral_ssids_factor_q(posdef::Bool, ptr::Ptr{Int64},
                                                   row::Ptr{Int32}, val::Ptr{Float128},
                                                   scale::Ptr{Float128}, akeep::Ptr{Cvoid},
                                                   fkeep::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{ssids_control_type{Float128,
                                                                                    Int32}},
                                                   inform::Ptr{ssids_inform_type{Float128,
                                                                                  Int32}})::Cvoid
end

function spral_ssids_factor(::Type{Float128}, ::Type{Int64}, posdef, ptr, row, val, scale,
                            akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple_64.spral_ssids_factor_q_64(posdef::Bool, ptr::Ptr{Int64},
                                                         row::Ptr{Int64},
                                                         val::Ptr{Float128},
                                                         scale::Ptr{Float128},
                                                         akeep::Ptr{Cvoid},
                                                         fkeep::Ptr{Ptr{Cvoid}},
                                                         control::Ptr{ssids_control_type{Float128,
                                                                                          Int64}},
                                                         inform::Ptr{ssids_inform_type{Float128,
                                                                                        Int64}})::Cvoid
end

export spral_ssids_factor_ptr32

function spral_ssids_factor_ptr32(::Type{Float32}, ::Type{Int32}, posdef, ptr, row, val,
                                  scale, akeep, fkeep, control, inform)
  @ccall libgalahad_single.spral_ssids_factor_ptr32_s(posdef::Bool, ptr::Ptr{Int32},
                                                      row::Ptr{Int32}, val::Ptr{Float32},
                                                      scale::Ptr{Float32},
                                                      akeep::Ptr{Cvoid},
                                                      fkeep::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{ssids_control_type{Float32,
                                                                                       Int32}},
                                                      inform::Ptr{ssids_inform_type{Float32,
                                                                                     Int32}})::Cvoid
end

function spral_ssids_factor_ptr32(::Type{Float32}, ::Type{Int64}, posdef, ptr, row, val,
                                  scale, akeep, fkeep, control, inform)
  @ccall libgalahad_single_64.spral_ssids_factor_ptr32_s_64(posdef::Bool, ptr::Ptr{Int64},
                                                            row::Ptr{Int64},
                                                            val::Ptr{Float32},
                                                            scale::Ptr{Float32},
                                                            akeep::Ptr{Cvoid},
                                                            fkeep::Ptr{Ptr{Cvoid}},
                                                            control::Ptr{ssids_control_type{Float32,
                                                                                             Int64}},
                                                            inform::Ptr{ssids_inform_type{Float32,
                                                                                           Int64}})::Cvoid
end

function spral_ssids_factor_ptr32(::Type{Float64}, ::Type{Int32}, posdef, ptr, row, val,
                                  scale, akeep, fkeep, control, inform)
  @ccall libgalahad_double.spral_ssids_factor_ptr32(posdef::Bool, ptr::Ptr{Int32},
                                                    row::Ptr{Int32}, val::Ptr{Float64},
                                                    scale::Ptr{Float64}, akeep::Ptr{Cvoid},
                                                    fkeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{ssids_control_type{Float64,
                                                                                     Int32}},
                                                    inform::Ptr{ssids_inform_type{Float64,
                                                                                   Int32}})::Cvoid
end

function spral_ssids_factor_ptr32(::Type{Float64}, ::Type{Int64}, posdef, ptr, row, val,
                                  scale, akeep, fkeep, control, inform)
  @ccall libgalahad_double_64.spral_ssids_factor_ptr32_64(posdef::Bool, ptr::Ptr{Int64},
                                                          row::Ptr{Int64},
                                                          val::Ptr{Float64},
                                                          scale::Ptr{Float64},
                                                          akeep::Ptr{Cvoid},
                                                          fkeep::Ptr{Ptr{Cvoid}},
                                                          control::Ptr{ssids_control_type{Float64,
                                                                                           Int64}},
                                                          inform::Ptr{ssids_inform_type{Float64,
                                                                                         Int64}})::Cvoid
end

function spral_ssids_factor_ptr32(::Type{Float128}, ::Type{Int32}, posdef, ptr, row, val,
                                  scale, akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple.spral_ssids_factor_ptr32_q(posdef::Bool, ptr::Ptr{Int32},
                                                         row::Ptr{Int32},
                                                         val::Ptr{Float128},
                                                         scale::Ptr{Float128},
                                                         akeep::Ptr{Cvoid},
                                                         fkeep::Ptr{Ptr{Cvoid}},
                                                         control::Ptr{ssids_control_type{Float128,
                                                                                          Int32}},
                                                         inform::Ptr{ssids_inform_type{Float128,
                                                                                        Int32}})::Cvoid
end

function spral_ssids_factor_ptr32(::Type{Float128}, ::Type{Int64}, posdef, ptr, row, val,
                                  scale, akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple_64.spral_ssids_factor_ptr32_q_64(posdef::Bool,
                                                               ptr::Ptr{Int64},
                                                               row::Ptr{Int64},
                                                               val::Ptr{Float128},
                                                               scale::Ptr{Float128},
                                                               akeep::Ptr{Cvoid},
                                                               fkeep::Ptr{Ptr{Cvoid}},
                                                               control::Ptr{ssids_control_type{Float128,
                                                                                                Int64}},
                                                               inform::Ptr{ssids_inform_type{Float128,
                                                                                              Int64}})::Cvoid
end

export spral_ssids_solve1

function spral_ssids_solve1(::Type{Float32}, ::Type{Int32}, job, x1, akeep, fkeep, control,
                            inform)
  @ccall libgalahad_single.spral_ssids_solve1_s(job::Int32, x1::Ptr{Float32},
                                                akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                control::Ptr{ssids_control_type{Float32,
                                                                                 Int32}},
                                                inform::Ptr{ssids_inform_type{Float32,
                                                                               Int32}})::Cvoid
end

function spral_ssids_solve1(::Type{Float32}, ::Type{Int64}, job, x1, akeep, fkeep, control,
                            inform)
  @ccall libgalahad_single_64.spral_ssids_solve1_s_64(job::Int64, x1::Ptr{Float32},
                                                      akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                      control::Ptr{ssids_control_type{Float32,
                                                                                       Int64}},
                                                      inform::Ptr{ssids_inform_type{Float32,
                                                                                     Int64}})::Cvoid
end

function spral_ssids_solve1(::Type{Float64}, ::Type{Int32}, job, x1, akeep, fkeep, control,
                            inform)
  @ccall libgalahad_double.spral_ssids_solve1(job::Int32, x1::Ptr{Float64},
                                              akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                              control::Ptr{ssids_control_type{Float64,
                                                                               Int32}},
                                              inform::Ptr{ssids_inform_type{Float64,Int32}})::Cvoid
end

function spral_ssids_solve1(::Type{Float64}, ::Type{Int64}, job, x1, akeep, fkeep, control,
                            inform)
  @ccall libgalahad_double_64.spral_ssids_solve1_64(job::Int64, x1::Ptr{Float64},
                                                    akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                    control::Ptr{ssids_control_type{Float64,
                                                                                     Int64}},
                                                    inform::Ptr{ssids_inform_type{Float64,
                                                                                   Int64}})::Cvoid
end

function spral_ssids_solve1(::Type{Float128}, ::Type{Int32}, job, x1, akeep, fkeep, control,
                            inform)
  @ccall libgalahad_quadruple.spral_ssids_solve1_q(job::Int32, x1::Ptr{Float128},
                                                   akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                   control::Ptr{ssids_control_type{Float128,
                                                                                    Int32}},
                                                   inform::Ptr{ssids_inform_type{Float128,
                                                                                  Int32}})::Cvoid
end

function spral_ssids_solve1(::Type{Float128}, ::Type{Int64}, job, x1, akeep, fkeep, control,
                            inform)
  @ccall libgalahad_quadruple_64.spral_ssids_solve1_q_64(job::Int64, x1::Ptr{Float128},
                                                         akeep::Ptr{Cvoid},
                                                         fkeep::Ptr{Cvoid},
                                                         control::Ptr{ssids_control_type{Float128,
                                                                                          Int64}},
                                                         inform::Ptr{ssids_inform_type{Float128,
                                                                                        Int64}})::Cvoid
end

export spral_ssids_solve

function spral_ssids_solve(::Type{Float32}, ::Type{Int32}, job, nrhs, x, ldx, akeep, fkeep,
                           control, inform)
  @ccall libgalahad_single.spral_ssids_solve_s(job::Int32, nrhs::Int32, x::Ptr{Float32},
                                               ldx::Int32, akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               control::Ptr{ssids_control_type{Float32,
                                                                                Int32}},
                                               inform::Ptr{ssids_inform_type{Float32,
                                                                              Int32}})::Cvoid
end

function spral_ssids_solve(::Type{Float32}, ::Type{Int64}, job, nrhs, x, ldx, akeep, fkeep,
                           control, inform)
  @ccall libgalahad_single_64.spral_ssids_solve_s_64(job::Int64, nrhs::Int64,
                                                     x::Ptr{Float32}, ldx::Int64,
                                                     akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                     control::Ptr{ssids_control_type{Float32,
                                                                                      Int64}},
                                                     inform::Ptr{ssids_inform_type{Float32,
                                                                                    Int64}})::Cvoid
end

function spral_ssids_solve(::Type{Float64}, ::Type{Int32}, job, nrhs, x, ldx, akeep, fkeep,
                           control, inform)
  @ccall libgalahad_double.spral_ssids_solve(job::Int32, nrhs::Int32, x::Ptr{Float64},
                                             ldx::Int32, akeep::Ptr{Cvoid},
                                             fkeep::Ptr{Cvoid},
                                             control::Ptr{ssids_control_type{Float64,
                                                                              Int32}},
                                             inform::Ptr{ssids_inform_type{Float64,Int32}})::Cvoid
end

function spral_ssids_solve(::Type{Float64}, ::Type{Int64}, job, nrhs, x, ldx, akeep, fkeep,
                           control, inform)
  @ccall libgalahad_double_64.spral_ssids_solve_64(job::Int64, nrhs::Int64, x::Ptr{Float64},
                                                   ldx::Int64, akeep::Ptr{Cvoid},
                                                   fkeep::Ptr{Cvoid},
                                                   control::Ptr{ssids_control_type{Float64,
                                                                                    Int64}},
                                                   inform::Ptr{ssids_inform_type{Float64,
                                                                                  Int64}})::Cvoid
end

function spral_ssids_solve(::Type{Float128}, ::Type{Int32}, job, nrhs, x, ldx, akeep, fkeep,
                           control, inform)
  @ccall libgalahad_quadruple.spral_ssids_solve_q(job::Int32, nrhs::Int32, x::Ptr{Float128},
                                                  ldx::Int32, akeep::Ptr{Cvoid},
                                                  fkeep::Ptr{Cvoid},
                                                  control::Ptr{ssids_control_type{Float128,
                                                                                   Int32}},
                                                  inform::Ptr{ssids_inform_type{Float128,
                                                                                 Int32}})::Cvoid
end

function spral_ssids_solve(::Type{Float128}, ::Type{Int64}, job, nrhs, x, ldx, akeep, fkeep,
                           control, inform)
  @ccall libgalahad_quadruple_64.spral_ssids_solve_q_64(job::Int64, nrhs::Int64,
                                                        x::Ptr{Float128}, ldx::Int64,
                                                        akeep::Ptr{Cvoid},
                                                        fkeep::Ptr{Cvoid},
                                                        control::Ptr{ssids_control_type{Float128,
                                                                                         Int64}},
                                                        inform::Ptr{ssids_inform_type{Float128,
                                                                                       Int64}})::Cvoid
end

export spral_ssids_free_akeep

function spral_ssids_free_akeep(::Type{Float32}, ::Type{Int32}, akeep)
  @ccall libgalahad_single.spral_ssids_free_akeep_s(akeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free_akeep(::Type{Float32}, ::Type{Int64}, akeep)
  @ccall libgalahad_single_64.spral_ssids_free_akeep_s_64(akeep::Ptr{Ptr{Cvoid}})::Int64
end

function spral_ssids_free_akeep(::Type{Float64}, ::Type{Int32}, akeep)
  @ccall libgalahad_double.spral_ssids_free_akeep(akeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free_akeep(::Type{Float64}, ::Type{Int64}, akeep)
  @ccall libgalahad_double_64.spral_ssids_free_akeep_64(akeep::Ptr{Ptr{Cvoid}})::Int64
end

function spral_ssids_free_akeep(::Type{Float128}, ::Type{Int32}, akeep)
  @ccall libgalahad_quadruple.spral_ssids_free_akeep_q(akeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free_akeep(::Type{Float128}, ::Type{Int64}, akeep)
  @ccall libgalahad_quadruple_64.spral_ssids_free_akeep_q_64(akeep::Ptr{Ptr{Cvoid}})::Int64
end

export spral_ssids_free_fkeep

function spral_ssids_free_fkeep(::Type{Float32}, ::Type{Int32}, fkeep)
  @ccall libgalahad_single.spral_ssids_free_fkeep_s(fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free_fkeep(::Type{Float32}, ::Type{Int64}, fkeep)
  @ccall libgalahad_single_64.spral_ssids_free_fkeep_s_64(fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function spral_ssids_free_fkeep(::Type{Float64}, ::Type{Int32}, fkeep)
  @ccall libgalahad_double.spral_ssids_free_fkeep(fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free_fkeep(::Type{Float64}, ::Type{Int64}, fkeep)
  @ccall libgalahad_double_64.spral_ssids_free_fkeep_64(fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function spral_ssids_free_fkeep(::Type{Float128}, ::Type{Int32}, fkeep)
  @ccall libgalahad_quadruple.spral_ssids_free_fkeep_q(fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free_fkeep(::Type{Float128}, ::Type{Int64}, fkeep)
  @ccall libgalahad_quadruple_64.spral_ssids_free_fkeep_q_64(fkeep::Ptr{Ptr{Cvoid}})::Int64
end

export spral_ssids_free

function spral_ssids_free(::Type{Float32}, ::Type{Int32}, akeep, fkeep)
  @ccall libgalahad_single.spral_ssids_free_s(akeep::Ptr{Ptr{Cvoid}},
                                              fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free(::Type{Float32}, ::Type{Int64}, akeep, fkeep)
  @ccall libgalahad_single_64.spral_ssids_free_s_64(akeep::Ptr{Ptr{Cvoid}},
                                                    fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function spral_ssids_free(::Type{Float64}, ::Type{Int32}, akeep, fkeep)
  @ccall libgalahad_double.spral_ssids_free(akeep::Ptr{Ptr{Cvoid}},
                                            fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free(::Type{Float64}, ::Type{Int64}, akeep, fkeep)
  @ccall libgalahad_double_64.spral_ssids_free_64(akeep::Ptr{Ptr{Cvoid}},
                                                  fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function spral_ssids_free(::Type{Float128}, ::Type{Int32}, akeep, fkeep)
  @ccall libgalahad_quadruple.spral_ssids_free_q(akeep::Ptr{Ptr{Cvoid}},
                                                 fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function spral_ssids_free(::Type{Float128}, ::Type{Int64}, akeep, fkeep)
  @ccall libgalahad_quadruple_64.spral_ssids_free_q_64(akeep::Ptr{Ptr{Cvoid}},
                                                       fkeep::Ptr{Ptr{Cvoid}})::Int64
end

export spral_ssids_enquire_posdef

function spral_ssids_enquire_posdef(::Type{Float32}, ::Type{Int32}, akeep, fkeep, control,
                                    inform, d)
  @ccall libgalahad_single.spral_ssids_enquire_posdef_s(akeep::Ptr{Cvoid},
                                                        fkeep::Ptr{Cvoid},
                                                        control::Ptr{ssids_control_type{Float32,
                                                                                         Int32}},
                                                        inform::Ptr{ssids_inform_type{Float32,
                                                                                       Int32}},
                                                        d::Ptr{Float32})::Cvoid
end

function spral_ssids_enquire_posdef(::Type{Float32}, ::Type{Int64}, akeep, fkeep, control,
                                    inform, d)
  @ccall libgalahad_single_64.spral_ssids_enquire_posdef_s_64(akeep::Ptr{Cvoid},
                                                              fkeep::Ptr{Cvoid},
                                                              control::Ptr{ssids_control_type{Float32,
                                                                                               Int64}},
                                                              inform::Ptr{ssids_inform_type{Float32,
                                                                                             Int64}},
                                                              d::Ptr{Float32})::Cvoid
end

function spral_ssids_enquire_posdef(::Type{Float64}, ::Type{Int32}, akeep, fkeep, control,
                                    inform, d)
  @ccall libgalahad_double.spral_ssids_enquire_posdef(akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                      control::Ptr{ssids_control_type{Float64,
                                                                                       Int32}},
                                                      inform::Ptr{ssids_inform_type{Float64,
                                                                                     Int32}},
                                                      d::Ptr{Float64})::Cvoid
end

function spral_ssids_enquire_posdef(::Type{Float64}, ::Type{Int64}, akeep, fkeep, control,
                                    inform, d)
  @ccall libgalahad_double_64.spral_ssids_enquire_posdef_64(akeep::Ptr{Cvoid},
                                                            fkeep::Ptr{Cvoid},
                                                            control::Ptr{ssids_control_type{Float64,
                                                                                             Int64}},
                                                            inform::Ptr{ssids_inform_type{Float64,
                                                                                           Int64}},
                                                            d::Ptr{Float64})::Cvoid
end

function spral_ssids_enquire_posdef(::Type{Float128}, ::Type{Int32}, akeep, fkeep, control,
                                    inform, d)
  @ccall libgalahad_quadruple.spral_ssids_enquire_posdef_q(akeep::Ptr{Cvoid},
                                                           fkeep::Ptr{Cvoid},
                                                           control::Ptr{ssids_control_type{Float128,
                                                                                            Int32}},
                                                           inform::Ptr{ssids_inform_type{Float128,
                                                                                          Int32}},
                                                           d::Ptr{Float128})::Cvoid
end

function spral_ssids_enquire_posdef(::Type{Float128}, ::Type{Int64}, akeep, fkeep, control,
                                    inform, d)
  @ccall libgalahad_quadruple_64.spral_ssids_enquire_posdef_q_64(akeep::Ptr{Cvoid},
                                                                 fkeep::Ptr{Cvoid},
                                                                 control::Ptr{ssids_control_type{Float128,
                                                                                                  Int64}},
                                                                 inform::Ptr{ssids_inform_type{Float128,
                                                                                                Int64}},
                                                                 d::Ptr{Float128})::Cvoid
end

export spral_ssids_enquire_indef

function spral_ssids_enquire_indef(::Type{Float32}, ::Type{Int32}, akeep, fkeep, control,
                                   inform, piv_order, d)
  @ccall libgalahad_single.spral_ssids_enquire_indef_s(akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                       control::Ptr{ssids_control_type{Float32,
                                                                                        Int32}},
                                                       inform::Ptr{ssids_inform_type{Float32,
                                                                                      Int32}},
                                                       piv_order::Ptr{Int32},
                                                       d::Ptr{Float32})::Cvoid
end

function spral_ssids_enquire_indef(::Type{Float32}, ::Type{Int64}, akeep, fkeep, control,
                                   inform, piv_order, d)
  @ccall libgalahad_single_64.spral_ssids_enquire_indef_s_64(akeep::Ptr{Cvoid},
                                                             fkeep::Ptr{Cvoid},
                                                             control::Ptr{ssids_control_type{Float32,
                                                                                              Int64}},
                                                             inform::Ptr{ssids_inform_type{Float32,
                                                                                            Int64}},
                                                             piv_order::Ptr{Int64},
                                                             d::Ptr{Float32})::Cvoid
end

function spral_ssids_enquire_indef(::Type{Float64}, ::Type{Int32}, akeep, fkeep, control,
                                   inform, piv_order, d)
  @ccall libgalahad_double.spral_ssids_enquire_indef(akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                     control::Ptr{ssids_control_type{Float64,
                                                                                      Int32}},
                                                     inform::Ptr{ssids_inform_type{Float64,
                                                                                    Int32}},
                                                     piv_order::Ptr{Int32},
                                                     d::Ptr{Float64})::Cvoid
end

function spral_ssids_enquire_indef(::Type{Float64}, ::Type{Int64}, akeep, fkeep, control,
                                   inform, piv_order, d)
  @ccall libgalahad_double_64.spral_ssids_enquire_indef_64(akeep::Ptr{Cvoid},
                                                           fkeep::Ptr{Cvoid},
                                                           control::Ptr{ssids_control_type{Float64,
                                                                                            Int64}},
                                                           inform::Ptr{ssids_inform_type{Float64,
                                                                                          Int64}},
                                                           piv_order::Ptr{Int64},
                                                           d::Ptr{Float64})::Cvoid
end

function spral_ssids_enquire_indef(::Type{Float128}, ::Type{Int32}, akeep, fkeep, control,
                                   inform, piv_order, d)
  @ccall libgalahad_quadruple.spral_ssids_enquire_indef_q(akeep::Ptr{Cvoid},
                                                          fkeep::Ptr{Cvoid},
                                                          control::Ptr{ssids_control_type{Float128,
                                                                                           Int32}},
                                                          inform::Ptr{ssids_inform_type{Float128,
                                                                                         Int32}},
                                                          piv_order::Ptr{Int32},
                                                          d::Ptr{Float128})::Cvoid
end

function spral_ssids_enquire_indef(::Type{Float128}, ::Type{Int64}, akeep, fkeep, control,
                                   inform, piv_order, d)
  @ccall libgalahad_quadruple_64.spral_ssids_enquire_indef_q_64(akeep::Ptr{Cvoid},
                                                                fkeep::Ptr{Cvoid},
                                                                control::Ptr{ssids_control_type{Float128,
                                                                                                 Int64}},
                                                                inform::Ptr{ssids_inform_type{Float128,
                                                                                               Int64}},
                                                                piv_order::Ptr{Int64},
                                                                d::Ptr{Float128})::Cvoid
end

export spral_ssids_alter

function spral_ssids_alter(::Type{Float32}, ::Type{Int32}, d, akeep, fkeep, control, inform)
  @ccall libgalahad_single.spral_ssids_alter_s(d::Ptr{Float32}, akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               control::Ptr{ssids_control_type{Float32,
                                                                                Int32}},
                                               inform::Ptr{ssids_inform_type{Float32,
                                                                              Int32}})::Cvoid
end

function spral_ssids_alter(::Type{Float32}, ::Type{Int64}, d, akeep, fkeep, control, inform)
  @ccall libgalahad_single_64.spral_ssids_alter_s_64(d::Ptr{Float32}, akeep::Ptr{Cvoid},
                                                     fkeep::Ptr{Cvoid},
                                                     control::Ptr{ssids_control_type{Float32,
                                                                                      Int64}},
                                                     inform::Ptr{ssids_inform_type{Float32,
                                                                                    Int64}})::Cvoid
end

function spral_ssids_alter(::Type{Float64}, ::Type{Int32}, d, akeep, fkeep, control, inform)
  @ccall libgalahad_double.spral_ssids_alter(d::Ptr{Float64}, akeep::Ptr{Cvoid},
                                             fkeep::Ptr{Cvoid},
                                             control::Ptr{ssids_control_type{Float64,
                                                                              Int32}},
                                             inform::Ptr{ssids_inform_type{Float64,Int32}})::Cvoid
end

function spral_ssids_alter(::Type{Float64}, ::Type{Int64}, d, akeep, fkeep, control, inform)
  @ccall libgalahad_double_64.spral_ssids_alter_64(d::Ptr{Float64}, akeep::Ptr{Cvoid},
                                                   fkeep::Ptr{Cvoid},
                                                   control::Ptr{ssids_control_type{Float64,
                                                                                    Int64}},
                                                   inform::Ptr{ssids_inform_type{Float64,
                                                                                  Int64}})::Cvoid
end

function spral_ssids_alter(::Type{Float128}, ::Type{Int32}, d, akeep, fkeep, control,
                           inform)
  @ccall libgalahad_quadruple.spral_ssids_alter_q(d::Ptr{Float128}, akeep::Ptr{Cvoid},
                                                  fkeep::Ptr{Cvoid},
                                                  control::Ptr{ssids_control_type{Float128,
                                                                                   Int32}},
                                                  inform::Ptr{ssids_inform_type{Float128,
                                                                                 Int32}})::Cvoid
end

function spral_ssids_alter(::Type{Float128}, ::Type{Int64}, d, akeep, fkeep, control,
                           inform)
  @ccall libgalahad_quadruple_64.spral_ssids_alter_q_64(d::Ptr{Float128}, akeep::Ptr{Cvoid},
                                                        fkeep::Ptr{Cvoid},
                                                        control::Ptr{ssids_control_type{Float128,
                                                                                         Int64}},
                                                        inform::Ptr{ssids_inform_type{Float128,
                                                                                       Int64}})::Cvoid
end
