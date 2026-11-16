export slblt_control_type

struct slblt_control_type{T,INT}
  array_base::INT
  print_level::INT
  unit_diagnostics::INT
  unit_error::INT
  unit_warning::INT
  ordering::INT
  nemin::INT
  ignore_numa::Bool
  max_load_inbalance::Float32
  scaling::INT
  small_subtree_threshold::Int64
  block_size::INT
  action::Bool
  pivot_method::INT
  small::T
  u::T
  nodend_control::nodend_control_type{INT}
  multiplier::T
  min_loadbalance::Float32
  failed_pivot_method::INT
end

export slblt_inform_type

struct slblt_inform_type{T,INT}
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
  nodend_inform::nodend_inform_type{T,INT}
  not_first_pass::INT
  not_second_pass::INT
  nparts::INT
  flops::Int64
end

export slblt_default_control

function slblt_default_control(::Type{Float32}, ::Type{Int32}, control)
  @ccall libgalahad_single.slblt_default_control_s(control::Ptr{slblt_control_type{Float32,
                                                                                   Int32}})::Cvoid
end

function slblt_default_control(::Type{Float32}, ::Type{Int64}, control)
  @ccall libgalahad_single_64.slblt_default_control_s_64(control::Ptr{slblt_control_type{Float32,
                                                                                         Int64}})::Cvoid
end

function slblt_default_control(::Type{Float64}, ::Type{Int32}, control)
  @ccall libgalahad_double.slblt_default_control(control::Ptr{slblt_control_type{Float64,
                                                                                 Int32}})::Cvoid
end

function slblt_default_control(::Type{Float64}, ::Type{Int64}, control)
  @ccall libgalahad_double_64.slblt_default_control_64(control::Ptr{slblt_control_type{Float64,
                                                                                       Int64}})::Cvoid
end

function slblt_default_control(::Type{Float128}, ::Type{Int32}, control)
  @ccall libgalahad_quadruple.slblt_default_control_q(control::Ptr{slblt_control_type{Float128,
                                                                                      Int32}})::Cvoid
end

function slblt_default_control(::Type{Float128}, ::Type{Int64}, control)
  @ccall libgalahad_quadruple_64.slblt_default_control_q_64(control::Ptr{slblt_control_type{Float128,
                                                                                            Int64}})::Cvoid
end

export slblt_analyse

function slblt_analyse(::Type{Float32}, ::Type{Int32}, check, n, order, ptr,
                       row, val, akeep, control, inform)
  @ccall libgalahad_single.slblt_analyse_s(check::Bool, n::Int32,
                                           order::Ptr{Int32}, ptr::Ptr{Int64},
                                           row::Ptr{Int32}, val::Ptr{Float32},
                                           akeep::Ptr{Ptr{Cvoid}},
                                           control::Ptr{slblt_control_type{Float32,
                                                                           Int32}},
                                           inform::Ptr{slblt_inform_type{Float32,
                                                                         Int32}})::Cvoid
end

function slblt_analyse(::Type{Float32}, ::Type{Int64}, check, n, order, ptr,
                       row, val, akeep, control, inform)
  @ccall libgalahad_single_64.slblt_analyse_s_64(check::Bool, n::Int64,
                                                 order::Ptr{Int64},
                                                 ptr::Ptr{Int64},
                                                 row::Ptr{Int64},
                                                 val::Ptr{Float32},
                                                 akeep::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{slblt_control_type{Float32,
                                                                                 Int64}},
                                                 inform::Ptr{slblt_inform_type{Float32,
                                                                               Int64}})::Cvoid
end

function slblt_analyse(::Type{Float64}, ::Type{Int32}, check, n, order, ptr,
                       row, val, akeep, control, inform)
  @ccall libgalahad_double.slblt_analyse(check::Bool, n::Int32,
                                         order::Ptr{Int32}, ptr::Ptr{Int64},
                                         row::Ptr{Int32}, val::Ptr{Float64},
                                         akeep::Ptr{Ptr{Cvoid}},
                                         control::Ptr{slblt_control_type{Float64,
                                                                         Int32}},
                                         inform::Ptr{slblt_inform_type{Float64,
                                                                       Int32}})::Cvoid
end

function slblt_analyse(::Type{Float64}, ::Type{Int64}, check, n, order, ptr,
                       row, val, akeep, control, inform)
  @ccall libgalahad_double_64.slblt_analyse_64(check::Bool, n::Int64,
                                               order::Ptr{Int64},
                                               ptr::Ptr{Int64}, row::Ptr{Int64},
                                               val::Ptr{Float64},
                                               akeep::Ptr{Ptr{Cvoid}},
                                               control::Ptr{slblt_control_type{Float64,
                                                                               Int64}},
                                               inform::Ptr{slblt_inform_type{Float64,
                                                                             Int64}})::Cvoid
end

function slblt_analyse(::Type{Float128}, ::Type{Int32}, check, n, order, ptr,
                       row, val, akeep, control, inform)
  @ccall libgalahad_quadruple.slblt_analyse_q(check::Bool, n::Int32,
                                              order::Ptr{Int32},
                                              ptr::Ptr{Int64}, row::Ptr{Int32},
                                              val::Ptr{Float128},
                                              akeep::Ptr{Ptr{Cvoid}},
                                              control::Ptr{slblt_control_type{Float128,
                                                                              Int32}},
                                              inform::Ptr{slblt_inform_type{Float128,
                                                                            Int32}})::Cvoid
end

function slblt_analyse(::Type{Float128}, ::Type{Int64}, check, n, order, ptr,
                       row, val, akeep, control, inform)
  @ccall libgalahad_quadruple_64.slblt_analyse_q_64(check::Bool, n::Int64,
                                                    order::Ptr{Int64},
                                                    ptr::Ptr{Int64},
                                                    row::Ptr{Int64},
                                                    val::Ptr{Float128},
                                                    akeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{slblt_control_type{Float128,
                                                                                    Int64}},
                                                    inform::Ptr{slblt_inform_type{Float128,
                                                                                  Int64}})::Cvoid
end

export slblt_analyse_ptr32

function slblt_analyse_ptr32(::Type{Float32}, ::Type{Int32}, check, n, order,
                             ptr, row, val, akeep, control, inform)
  @ccall libgalahad_single.slblt_analyse_ptr32_s(check::Bool, n::Int32,
                                                 order::Ptr{Int32},
                                                 ptr::Ptr{Int32},
                                                 row::Ptr{Int32},
                                                 val::Ptr{Float32},
                                                 akeep::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{slblt_control_type{Float32,
                                                                                 Int32}},
                                                 inform::Ptr{slblt_inform_type{Float32,
                                                                               Int32}})::Cvoid
end

function slblt_analyse_ptr32(::Type{Float32}, ::Type{Int64}, check, n, order,
                             ptr, row, val, akeep, control, inform)
  @ccall libgalahad_single_64.slblt_analyse_ptr32_s_64(check::Bool, n::Int64,
                                                       order::Ptr{Int64},
                                                       ptr::Ptr{Int64},
                                                       row::Ptr{Int64},
                                                       val::Ptr{Float32},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       control::Ptr{slblt_control_type{Float32,
                                                                                       Int64}},
                                                       inform::Ptr{slblt_inform_type{Float32,
                                                                                     Int64}})::Cvoid
end

function slblt_analyse_ptr32(::Type{Float64}, ::Type{Int32}, check, n, order,
                             ptr, row, val, akeep, control, inform)
  @ccall libgalahad_double.slblt_analyse_ptr32(check::Bool, n::Int32,
                                               order::Ptr{Int32},
                                               ptr::Ptr{Int32}, row::Ptr{Int32},
                                               val::Ptr{Float64},
                                               akeep::Ptr{Ptr{Cvoid}},
                                               control::Ptr{slblt_control_type{Float64,
                                                                               Int32}},
                                               inform::Ptr{slblt_inform_type{Float64,
                                                                             Int32}})::Cvoid
end

function slblt_analyse_ptr32(::Type{Float64}, ::Type{Int64}, check, n, order,
                             ptr, row, val, akeep, control, inform)
  @ccall libgalahad_double_64.slblt_analyse_ptr32_64(check::Bool, n::Int64,
                                                     order::Ptr{Int64},
                                                     ptr::Ptr{Int64},
                                                     row::Ptr{Int64},
                                                     val::Ptr{Float64},
                                                     akeep::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{slblt_control_type{Float64,
                                                                                     Int64}},
                                                     inform::Ptr{slblt_inform_type{Float64,
                                                                                   Int64}})::Cvoid
end

function slblt_analyse_ptr32(::Type{Float128}, ::Type{Int32}, check, n, order,
                             ptr, row, val, akeep, control, inform)
  @ccall libgalahad_quadruple.slblt_analyse_ptr32_q(check::Bool, n::Int32,
                                                    order::Ptr{Int32},
                                                    ptr::Ptr{Int32},
                                                    row::Ptr{Int32},
                                                    val::Ptr{Float128},
                                                    akeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{slblt_control_type{Float128,
                                                                                    Int32}},
                                                    inform::Ptr{slblt_inform_type{Float128,
                                                                                  Int32}})::Cvoid
end

function slblt_analyse_ptr32(::Type{Float128}, ::Type{Int64}, check, n, order,
                             ptr, row, val, akeep, control, inform)
  @ccall libgalahad_quadruple_64.slblt_analyse_ptr32_q_64(check::Bool, n::Int64,
                                                          order::Ptr{Int64},
                                                          ptr::Ptr{Int64},
                                                          row::Ptr{Int64},
                                                          val::Ptr{Float128},
                                                          akeep::Ptr{Ptr{Cvoid}},
                                                          control::Ptr{slblt_control_type{Float128,
                                                                                          Int64}},
                                                          inform::Ptr{slblt_inform_type{Float128,
                                                                                        Int64}})::Cvoid
end

export slblt_analyse_coord

function slblt_analyse_coord(::Type{Float32}, ::Type{Int32}, n, order, ne, row,
                             col, val, akeep, control, inform)
  @ccall libgalahad_single.slblt_analyse_coord_s(n::Int32, order::Ptr{Int32},
                                                 ne::Int64, row::Ptr{Int32},
                                                 col::Ptr{Int32},
                                                 val::Ptr{Float32},
                                                 akeep::Ptr{Ptr{Cvoid}},
                                                 control::Ptr{slblt_control_type{Float32,
                                                                                 Int32}},
                                                 inform::Ptr{slblt_inform_type{Float32,
                                                                               Int32}})::Cvoid
end

function slblt_analyse_coord(::Type{Float32}, ::Type{Int64}, n, order, ne, row,
                             col, val, akeep, control, inform)
  @ccall libgalahad_single_64.slblt_analyse_coord_s_64(n::Int64,
                                                       order::Ptr{Int64},
                                                       ne::Int64,
                                                       row::Ptr{Int64},
                                                       col::Ptr{Int64},
                                                       val::Ptr{Float32},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       control::Ptr{slblt_control_type{Float32,
                                                                                       Int64}},
                                                       inform::Ptr{slblt_inform_type{Float32,
                                                                                     Int64}})::Cvoid
end

function slblt_analyse_coord(::Type{Float64}, ::Type{Int32}, n, order, ne, row,
                             col, val, akeep, control, inform)
  @ccall libgalahad_double.slblt_analyse_coord(n::Int32, order::Ptr{Int32},
                                               ne::Int64, row::Ptr{Int32},
                                               col::Ptr{Int32},
                                               val::Ptr{Float64},
                                               akeep::Ptr{Ptr{Cvoid}},
                                               control::Ptr{slblt_control_type{Float64,
                                                                               Int32}},
                                               inform::Ptr{slblt_inform_type{Float64,
                                                                             Int32}})::Cvoid
end

function slblt_analyse_coord(::Type{Float64}, ::Type{Int64}, n, order, ne, row,
                             col, val, akeep, control, inform)
  @ccall libgalahad_double_64.slblt_analyse_coord_64(n::Int64,
                                                     order::Ptr{Int64},
                                                     ne::Int64, row::Ptr{Int64},
                                                     col::Ptr{Int64},
                                                     val::Ptr{Float64},
                                                     akeep::Ptr{Ptr{Cvoid}},
                                                     control::Ptr{slblt_control_type{Float64,
                                                                                     Int64}},
                                                     inform::Ptr{slblt_inform_type{Float64,
                                                                                   Int64}})::Cvoid
end

function slblt_analyse_coord(::Type{Float128}, ::Type{Int32}, n, order, ne, row,
                             col, val, akeep, control, inform)
  @ccall libgalahad_quadruple.slblt_analyse_coord_q(n::Int32, order::Ptr{Int32},
                                                    ne::Int64, row::Ptr{Int32},
                                                    col::Ptr{Int32},
                                                    val::Ptr{Float128},
                                                    akeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{slblt_control_type{Float128,
                                                                                    Int32}},
                                                    inform::Ptr{slblt_inform_type{Float128,
                                                                                  Int32}})::Cvoid
end

function slblt_analyse_coord(::Type{Float128}, ::Type{Int64}, n, order, ne, row,
                             col, val, akeep, control, inform)
  @ccall libgalahad_quadruple_64.slblt_analyse_coord_q_64(n::Int64,
                                                          order::Ptr{Int64},
                                                          ne::Int64,
                                                          row::Ptr{Int64},
                                                          col::Ptr{Int64},
                                                          val::Ptr{Float128},
                                                          akeep::Ptr{Ptr{Cvoid}},
                                                          control::Ptr{slblt_control_type{Float128,
                                                                                          Int64}},
                                                          inform::Ptr{slblt_inform_type{Float128,
                                                                                        Int64}})::Cvoid
end

export slblt_factor

function slblt_factor(::Type{Float32}, ::Type{Int32}, posdef, ptr, row, val,
                      scale, akeep, fkeep, control, inform)
  @ccall libgalahad_single.slblt_factor_s(posdef::Bool, ptr::Ptr{Int64},
                                          row::Ptr{Int32}, val::Ptr{Float32},
                                          scale::Ptr{Float32},
                                          akeep::Ptr{Cvoid},
                                          fkeep::Ptr{Ptr{Cvoid}},
                                          control::Ptr{slblt_control_type{Float32,
                                                                          Int32}},
                                          inform::Ptr{slblt_inform_type{Float32,
                                                                        Int32}})::Cvoid
end

function slblt_factor(::Type{Float32}, ::Type{Int64}, posdef, ptr, row, val,
                      scale, akeep, fkeep, control, inform)
  @ccall libgalahad_single_64.slblt_factor_s_64(posdef::Bool, ptr::Ptr{Int64},
                                                row::Ptr{Int64},
                                                val::Ptr{Float32},
                                                scale::Ptr{Float32},
                                                akeep::Ptr{Cvoid},
                                                fkeep::Ptr{Ptr{Cvoid}},
                                                control::Ptr{slblt_control_type{Float32,
                                                                                Int64}},
                                                inform::Ptr{slblt_inform_type{Float32,
                                                                              Int64}})::Cvoid
end

function slblt_factor(::Type{Float64}, ::Type{Int32}, posdef, ptr, row, val,
                      scale, akeep, fkeep, control, inform)
  @ccall libgalahad_double.slblt_factor(posdef::Bool, ptr::Ptr{Int64},
                                        row::Ptr{Int32}, val::Ptr{Float64},
                                        scale::Ptr{Float64}, akeep::Ptr{Cvoid},
                                        fkeep::Ptr{Ptr{Cvoid}},
                                        control::Ptr{slblt_control_type{Float64,
                                                                        Int32}},
                                        inform::Ptr{slblt_inform_type{Float64,
                                                                      Int32}})::Cvoid
end

function slblt_factor(::Type{Float64}, ::Type{Int64}, posdef, ptr, row, val,
                      scale, akeep, fkeep, control, inform)
  @ccall libgalahad_double_64.slblt_factor_64(posdef::Bool, ptr::Ptr{Int64},
                                              row::Ptr{Int64},
                                              val::Ptr{Float64},
                                              scale::Ptr{Float64},
                                              akeep::Ptr{Cvoid},
                                              fkeep::Ptr{Ptr{Cvoid}},
                                              control::Ptr{slblt_control_type{Float64,
                                                                              Int64}},
                                              inform::Ptr{slblt_inform_type{Float64,
                                                                            Int64}})::Cvoid
end

function slblt_factor(::Type{Float128}, ::Type{Int32}, posdef, ptr, row, val,
                      scale, akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple.slblt_factor_q(posdef::Bool, ptr::Ptr{Int64},
                                             row::Ptr{Int32},
                                             val::Ptr{Float128},
                                             scale::Ptr{Float128},
                                             akeep::Ptr{Cvoid},
                                             fkeep::Ptr{Ptr{Cvoid}},
                                             control::Ptr{slblt_control_type{Float128,
                                                                             Int32}},
                                             inform::Ptr{slblt_inform_type{Float128,
                                                                           Int32}})::Cvoid
end

function slblt_factor(::Type{Float128}, ::Type{Int64}, posdef, ptr, row, val,
                      scale, akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple_64.slblt_factor_q_64(posdef::Bool,
                                                   ptr::Ptr{Int64},
                                                   row::Ptr{Int64},
                                                   val::Ptr{Float128},
                                                   scale::Ptr{Float128},
                                                   akeep::Ptr{Cvoid},
                                                   fkeep::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{slblt_control_type{Float128,
                                                                                   Int64}},
                                                   inform::Ptr{slblt_inform_type{Float128,
                                                                                 Int64}})::Cvoid
end

export slblt_factor_ptr32

function slblt_factor_ptr32(::Type{Float32}, ::Type{Int32}, posdef, ptr, row,
                            val, scale, akeep, fkeep, control, inform)
  @ccall libgalahad_single.slblt_factor_ptr32_s(posdef::Bool, ptr::Ptr{Int32},
                                                row::Ptr{Int32},
                                                val::Ptr{Float32},
                                                scale::Ptr{Float32},
                                                akeep::Ptr{Cvoid},
                                                fkeep::Ptr{Ptr{Cvoid}},
                                                control::Ptr{slblt_control_type{Float32,
                                                                                Int32}},
                                                inform::Ptr{slblt_inform_type{Float32,
                                                                              Int32}})::Cvoid
end

function slblt_factor_ptr32(::Type{Float32}, ::Type{Int64}, posdef, ptr, row,
                            val, scale, akeep, fkeep, control, inform)
  @ccall libgalahad_single_64.slblt_factor_ptr32_s_64(posdef::Bool,
                                                      ptr::Ptr{Int64},
                                                      row::Ptr{Int64},
                                                      val::Ptr{Float32},
                                                      scale::Ptr{Float32},
                                                      akeep::Ptr{Cvoid},
                                                      fkeep::Ptr{Ptr{Cvoid}},
                                                      control::Ptr{slblt_control_type{Float32,
                                                                                      Int64}},
                                                      inform::Ptr{slblt_inform_type{Float32,
                                                                                    Int64}})::Cvoid
end

function slblt_factor_ptr32(::Type{Float64}, ::Type{Int32}, posdef, ptr, row,
                            val, scale, akeep, fkeep, control, inform)
  @ccall libgalahad_double.slblt_factor_ptr32(posdef::Bool, ptr::Ptr{Int32},
                                              row::Ptr{Int32},
                                              val::Ptr{Float64},
                                              scale::Ptr{Float64},
                                              akeep::Ptr{Cvoid},
                                              fkeep::Ptr{Ptr{Cvoid}},
                                              control::Ptr{slblt_control_type{Float64,
                                                                              Int32}},
                                              inform::Ptr{slblt_inform_type{Float64,
                                                                            Int32}})::Cvoid
end

function slblt_factor_ptr32(::Type{Float64}, ::Type{Int64}, posdef, ptr, row,
                            val, scale, akeep, fkeep, control, inform)
  @ccall libgalahad_double_64.slblt_factor_ptr32_64(posdef::Bool,
                                                    ptr::Ptr{Int64},
                                                    row::Ptr{Int64},
                                                    val::Ptr{Float64},
                                                    scale::Ptr{Float64},
                                                    akeep::Ptr{Cvoid},
                                                    fkeep::Ptr{Ptr{Cvoid}},
                                                    control::Ptr{slblt_control_type{Float64,
                                                                                    Int64}},
                                                    inform::Ptr{slblt_inform_type{Float64,
                                                                                  Int64}})::Cvoid
end

function slblt_factor_ptr32(::Type{Float128}, ::Type{Int32}, posdef, ptr, row,
                            val, scale, akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple.slblt_factor_ptr32_q(posdef::Bool,
                                                   ptr::Ptr{Int32},
                                                   row::Ptr{Int32},
                                                   val::Ptr{Float128},
                                                   scale::Ptr{Float128},
                                                   akeep::Ptr{Cvoid},
                                                   fkeep::Ptr{Ptr{Cvoid}},
                                                   control::Ptr{slblt_control_type{Float128,
                                                                                   Int32}},
                                                   inform::Ptr{slblt_inform_type{Float128,
                                                                                 Int32}})::Cvoid
end

function slblt_factor_ptr32(::Type{Float128}, ::Type{Int64}, posdef, ptr, row,
                            val, scale, akeep, fkeep, control, inform)
  @ccall libgalahad_quadruple_64.slblt_factor_ptr32_q_64(posdef::Bool,
                                                         ptr::Ptr{Int64},
                                                         row::Ptr{Int64},
                                                         val::Ptr{Float128},
                                                         scale::Ptr{Float128},
                                                         akeep::Ptr{Cvoid},
                                                         fkeep::Ptr{Ptr{Cvoid}},
                                                         control::Ptr{slblt_control_type{Float128,
                                                                                         Int64}},
                                                         inform::Ptr{slblt_inform_type{Float128,
                                                                                       Int64}})::Cvoid
end

export slblt_solve1

function slblt_solve1(::Type{Float32}, ::Type{Int32}, job, x1, akeep, fkeep,
                      control, inform)
  @ccall libgalahad_single.slblt_solve1_s(job::Int32, x1::Ptr{Float32},
                                          akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                          control::Ptr{slblt_control_type{Float32,
                                                                          Int32}},
                                          inform::Ptr{slblt_inform_type{Float32,
                                                                        Int32}})::Cvoid
end

function slblt_solve1(::Type{Float32}, ::Type{Int64}, job, x1, akeep, fkeep,
                      control, inform)
  @ccall libgalahad_single_64.slblt_solve1_s_64(job::Int64, x1::Ptr{Float32},
                                                akeep::Ptr{Cvoid},
                                                fkeep::Ptr{Cvoid},
                                                control::Ptr{slblt_control_type{Float32,
                                                                                Int64}},
                                                inform::Ptr{slblt_inform_type{Float32,
                                                                              Int64}})::Cvoid
end

function slblt_solve1(::Type{Float64}, ::Type{Int32}, job, x1, akeep, fkeep,
                      control, inform)
  @ccall libgalahad_double.slblt_solve1(job::Int32, x1::Ptr{Float64},
                                        akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                        control::Ptr{slblt_control_type{Float64,
                                                                        Int32}},
                                        inform::Ptr{slblt_inform_type{Float64,
                                                                      Int32}})::Cvoid
end

function slblt_solve1(::Type{Float64}, ::Type{Int64}, job, x1, akeep, fkeep,
                      control, inform)
  @ccall libgalahad_double_64.slblt_solve1_64(job::Int64, x1::Ptr{Float64},
                                              akeep::Ptr{Cvoid},
                                              fkeep::Ptr{Cvoid},
                                              control::Ptr{slblt_control_type{Float64,
                                                                              Int64}},
                                              inform::Ptr{slblt_inform_type{Float64,
                                                                            Int64}})::Cvoid
end

function slblt_solve1(::Type{Float128}, ::Type{Int32}, job, x1, akeep, fkeep,
                      control, inform)
  @ccall libgalahad_quadruple.slblt_solve1_q(job::Int32, x1::Ptr{Float128},
                                             akeep::Ptr{Cvoid},
                                             fkeep::Ptr{Cvoid},
                                             control::Ptr{slblt_control_type{Float128,
                                                                             Int32}},
                                             inform::Ptr{slblt_inform_type{Float128,
                                                                           Int32}})::Cvoid
end

function slblt_solve1(::Type{Float128}, ::Type{Int64}, job, x1, akeep, fkeep,
                      control, inform)
  @ccall libgalahad_quadruple_64.slblt_solve1_q_64(job::Int64,
                                                   x1::Ptr{Float128},
                                                   akeep::Ptr{Cvoid},
                                                   fkeep::Ptr{Cvoid},
                                                   control::Ptr{slblt_control_type{Float128,
                                                                                   Int64}},
                                                   inform::Ptr{slblt_inform_type{Float128,
                                                                                 Int64}})::Cvoid
end

export slblt_solve

function slblt_solve(::Type{Float32}, ::Type{Int32}, job, nrhs, x, ldx, akeep,
                     fkeep, control, inform)
  @ccall libgalahad_single.slblt_solve_s(job::Int32, nrhs::Int32,
                                         x::Ptr{Float32}, ldx::Int32,
                                         akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                         control::Ptr{slblt_control_type{Float32,
                                                                         Int32}},
                                         inform::Ptr{slblt_inform_type{Float32,
                                                                       Int32}})::Cvoid
end

function slblt_solve(::Type{Float32}, ::Type{Int64}, job, nrhs, x, ldx, akeep,
                     fkeep, control, inform)
  @ccall libgalahad_single_64.slblt_solve_s_64(job::Int64, nrhs::Int64,
                                               x::Ptr{Float32}, ldx::Int64,
                                               akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               control::Ptr{slblt_control_type{Float32,
                                                                               Int64}},
                                               inform::Ptr{slblt_inform_type{Float32,
                                                                             Int64}})::Cvoid
end

function slblt_solve(::Type{Float64}, ::Type{Int32}, job, nrhs, x, ldx, akeep,
                     fkeep, control, inform)
  @ccall libgalahad_double.slblt_solve(job::Int32, nrhs::Int32, x::Ptr{Float64},
                                       ldx::Int32, akeep::Ptr{Cvoid},
                                       fkeep::Ptr{Cvoid},
                                       control::Ptr{slblt_control_type{Float64,
                                                                       Int32}},
                                       inform::Ptr{slblt_inform_type{Float64,
                                                                     Int32}})::Cvoid
end

function slblt_solve(::Type{Float64}, ::Type{Int64}, job, nrhs, x, ldx, akeep,
                     fkeep, control, inform)
  @ccall libgalahad_double_64.slblt_solve_64(job::Int64, nrhs::Int64,
                                             x::Ptr{Float64}, ldx::Int64,
                                             akeep::Ptr{Cvoid},
                                             fkeep::Ptr{Cvoid},
                                             control::Ptr{slblt_control_type{Float64,
                                                                             Int64}},
                                             inform::Ptr{slblt_inform_type{Float64,
                                                                           Int64}})::Cvoid
end

function slblt_solve(::Type{Float128}, ::Type{Int32}, job, nrhs, x, ldx, akeep,
                     fkeep, control, inform)
  @ccall libgalahad_quadruple.slblt_solve_q(job::Int32, nrhs::Int32,
                                            x::Ptr{Float128}, ldx::Int32,
                                            akeep::Ptr{Cvoid},
                                            fkeep::Ptr{Cvoid},
                                            control::Ptr{slblt_control_type{Float128,
                                                                            Int32}},
                                            inform::Ptr{slblt_inform_type{Float128,
                                                                          Int32}})::Cvoid
end

function slblt_solve(::Type{Float128}, ::Type{Int64}, job, nrhs, x, ldx, akeep,
                     fkeep, control, inform)
  @ccall libgalahad_quadruple_64.slblt_solve_q_64(job::Int64, nrhs::Int64,
                                                  x::Ptr{Float128}, ldx::Int64,
                                                  akeep::Ptr{Cvoid},
                                                  fkeep::Ptr{Cvoid},
                                                  control::Ptr{slblt_control_type{Float128,
                                                                                  Int64}},
                                                  inform::Ptr{slblt_inform_type{Float128,
                                                                                Int64}})::Cvoid
end

export slblt_free_akeep

function slblt_free_akeep(::Type{Float32}, ::Type{Int32}, akeep)
  @ccall libgalahad_single.slblt_free_akeep_s(akeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free_akeep(::Type{Float32}, ::Type{Int64}, akeep)
  @ccall libgalahad_single_64.slblt_free_akeep_s_64(akeep::Ptr{Ptr{Cvoid}})::Int64
end

function slblt_free_akeep(::Type{Float64}, ::Type{Int32}, akeep)
  @ccall libgalahad_double.slblt_free_akeep(akeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free_akeep(::Type{Float64}, ::Type{Int64}, akeep)
  @ccall libgalahad_double_64.slblt_free_akeep_64(akeep::Ptr{Ptr{Cvoid}})::Int64
end

function slblt_free_akeep(::Type{Float128}, ::Type{Int32}, akeep)
  @ccall libgalahad_quadruple.slblt_free_akeep_q(akeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free_akeep(::Type{Float128}, ::Type{Int64}, akeep)
  @ccall libgalahad_quadruple_64.slblt_free_akeep_q_64(akeep::Ptr{Ptr{Cvoid}})::Int64
end

export slblt_free_fkeep

function slblt_free_fkeep(::Type{Float32}, ::Type{Int32}, fkeep)
  @ccall libgalahad_single.slblt_free_fkeep_s(fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free_fkeep(::Type{Float32}, ::Type{Int64}, fkeep)
  @ccall libgalahad_single_64.slblt_free_fkeep_s_64(fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function slblt_free_fkeep(::Type{Float64}, ::Type{Int32}, fkeep)
  @ccall libgalahad_double.slblt_free_fkeep(fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free_fkeep(::Type{Float64}, ::Type{Int64}, fkeep)
  @ccall libgalahad_double_64.slblt_free_fkeep_64(fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function slblt_free_fkeep(::Type{Float128}, ::Type{Int32}, fkeep)
  @ccall libgalahad_quadruple.slblt_free_fkeep_q(fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free_fkeep(::Type{Float128}, ::Type{Int64}, fkeep)
  @ccall libgalahad_quadruple_64.slblt_free_fkeep_q_64(fkeep::Ptr{Ptr{Cvoid}})::Int64
end

export slblt_free

function slblt_free(::Type{Float32}, ::Type{Int32}, akeep, fkeep)
  @ccall libgalahad_single.slblt_free_s(akeep::Ptr{Ptr{Cvoid}},
                                        fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free(::Type{Float32}, ::Type{Int64}, akeep, fkeep)
  @ccall libgalahad_single_64.slblt_free_s_64(akeep::Ptr{Ptr{Cvoid}},
                                              fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function slblt_free(::Type{Float64}, ::Type{Int32}, akeep, fkeep)
  @ccall libgalahad_double.slblt_free(akeep::Ptr{Ptr{Cvoid}},
                                      fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free(::Type{Float64}, ::Type{Int64}, akeep, fkeep)
  @ccall libgalahad_double_64.slblt_free_64(akeep::Ptr{Ptr{Cvoid}},
                                            fkeep::Ptr{Ptr{Cvoid}})::Int64
end

function slblt_free(::Type{Float128}, ::Type{Int32}, akeep, fkeep)
  @ccall libgalahad_quadruple.slblt_free_q(akeep::Ptr{Ptr{Cvoid}},
                                           fkeep::Ptr{Ptr{Cvoid}})::Int32
end

function slblt_free(::Type{Float128}, ::Type{Int64}, akeep, fkeep)
  @ccall libgalahad_quadruple_64.slblt_free_q_64(akeep::Ptr{Ptr{Cvoid}},
                                                 fkeep::Ptr{Ptr{Cvoid}})::Int64
end

export slblt_enquire_posdef

function slblt_enquire_posdef(::Type{Float32}, ::Type{Int32}, akeep, fkeep,
                              control, inform, d)
  @ccall libgalahad_single.slblt_enquire_posdef_s(akeep::Ptr{Cvoid},
                                                  fkeep::Ptr{Cvoid},
                                                  control::Ptr{slblt_control_type{Float32,
                                                                                  Int32}},
                                                  inform::Ptr{slblt_inform_type{Float32,
                                                                                Int32}},
                                                  d::Ptr{Float32})::Cvoid
end

function slblt_enquire_posdef(::Type{Float32}, ::Type{Int64}, akeep, fkeep,
                              control, inform, d)
  @ccall libgalahad_single_64.slblt_enquire_posdef_s_64(akeep::Ptr{Cvoid},
                                                        fkeep::Ptr{Cvoid},
                                                        control::Ptr{slblt_control_type{Float32,
                                                                                        Int64}},
                                                        inform::Ptr{slblt_inform_type{Float32,
                                                                                      Int64}},
                                                        d::Ptr{Float32})::Cvoid
end

function slblt_enquire_posdef(::Type{Float64}, ::Type{Int32}, akeep, fkeep,
                              control, inform, d)
  @ccall libgalahad_double.slblt_enquire_posdef(akeep::Ptr{Cvoid},
                                                fkeep::Ptr{Cvoid},
                                                control::Ptr{slblt_control_type{Float64,
                                                                                Int32}},
                                                inform::Ptr{slblt_inform_type{Float64,
                                                                              Int32}},
                                                d::Ptr{Float64})::Cvoid
end

function slblt_enquire_posdef(::Type{Float64}, ::Type{Int64}, akeep, fkeep,
                              control, inform, d)
  @ccall libgalahad_double_64.slblt_enquire_posdef_64(akeep::Ptr{Cvoid},
                                                      fkeep::Ptr{Cvoid},
                                                      control::Ptr{slblt_control_type{Float64,
                                                                                      Int64}},
                                                      inform::Ptr{slblt_inform_type{Float64,
                                                                                    Int64}},
                                                      d::Ptr{Float64})::Cvoid
end

function slblt_enquire_posdef(::Type{Float128}, ::Type{Int32}, akeep, fkeep,
                              control, inform, d)
  @ccall libgalahad_quadruple.slblt_enquire_posdef_q(akeep::Ptr{Cvoid},
                                                     fkeep::Ptr{Cvoid},
                                                     control::Ptr{slblt_control_type{Float128,
                                                                                     Int32}},
                                                     inform::Ptr{slblt_inform_type{Float128,
                                                                                   Int32}},
                                                     d::Ptr{Float128})::Cvoid
end

function slblt_enquire_posdef(::Type{Float128}, ::Type{Int64}, akeep, fkeep,
                              control, inform, d)
  @ccall libgalahad_quadruple_64.slblt_enquire_posdef_q_64(akeep::Ptr{Cvoid},
                                                           fkeep::Ptr{Cvoid},
                                                           control::Ptr{slblt_control_type{Float128,
                                                                                           Int64}},
                                                           inform::Ptr{slblt_inform_type{Float128,
                                                                                         Int64}},
                                                           d::Ptr{Float128})::Cvoid
end

export slblt_enquire_indef

function slblt_enquire_indef(::Type{Float32}, ::Type{Int32}, akeep, fkeep,
                             control, inform, piv_order, d)
  @ccall libgalahad_single.slblt_enquire_indef_s(akeep::Ptr{Cvoid},
                                                 fkeep::Ptr{Cvoid},
                                                 control::Ptr{slblt_control_type{Float32,
                                                                                 Int32}},
                                                 inform::Ptr{slblt_inform_type{Float32,
                                                                               Int32}},
                                                 piv_order::Ptr{Int32},
                                                 d::Ptr{Float32})::Cvoid
end

function slblt_enquire_indef(::Type{Float32}, ::Type{Int64}, akeep, fkeep,
                             control, inform, piv_order, d)
  @ccall libgalahad_single_64.slblt_enquire_indef_s_64(akeep::Ptr{Cvoid},
                                                       fkeep::Ptr{Cvoid},
                                                       control::Ptr{slblt_control_type{Float32,
                                                                                       Int64}},
                                                       inform::Ptr{slblt_inform_type{Float32,
                                                                                     Int64}},
                                                       piv_order::Ptr{Int64},
                                                       d::Ptr{Float32})::Cvoid
end

function slblt_enquire_indef(::Type{Float64}, ::Type{Int32}, akeep, fkeep,
                             control, inform, piv_order, d)
  @ccall libgalahad_double.slblt_enquire_indef(akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               control::Ptr{slblt_control_type{Float64,
                                                                               Int32}},
                                               inform::Ptr{slblt_inform_type{Float64,
                                                                             Int32}},
                                               piv_order::Ptr{Int32},
                                               d::Ptr{Float64})::Cvoid
end

function slblt_enquire_indef(::Type{Float64}, ::Type{Int64}, akeep, fkeep,
                             control, inform, piv_order, d)
  @ccall libgalahad_double_64.slblt_enquire_indef_64(akeep::Ptr{Cvoid},
                                                     fkeep::Ptr{Cvoid},
                                                     control::Ptr{slblt_control_type{Float64,
                                                                                     Int64}},
                                                     inform::Ptr{slblt_inform_type{Float64,
                                                                                   Int64}},
                                                     piv_order::Ptr{Int64},
                                                     d::Ptr{Float64})::Cvoid
end

function slblt_enquire_indef(::Type{Float128}, ::Type{Int32}, akeep, fkeep,
                             control, inform, piv_order, d)
  @ccall libgalahad_quadruple.slblt_enquire_indef_q(akeep::Ptr{Cvoid},
                                                    fkeep::Ptr{Cvoid},
                                                    control::Ptr{slblt_control_type{Float128,
                                                                                    Int32}},
                                                    inform::Ptr{slblt_inform_type{Float128,
                                                                                  Int32}},
                                                    piv_order::Ptr{Int32},
                                                    d::Ptr{Float128})::Cvoid
end

function slblt_enquire_indef(::Type{Float128}, ::Type{Int64}, akeep, fkeep,
                             control, inform, piv_order, d)
  @ccall libgalahad_quadruple_64.slblt_enquire_indef_q_64(akeep::Ptr{Cvoid},
                                                          fkeep::Ptr{Cvoid},
                                                          control::Ptr{slblt_control_type{Float128,
                                                                                          Int64}},
                                                          inform::Ptr{slblt_inform_type{Float128,
                                                                                        Int64}},
                                                          piv_order::Ptr{Int64},
                                                          d::Ptr{Float128})::Cvoid
end

export slblt_alter

function slblt_alter(::Type{Float32}, ::Type{Int32}, d, akeep, fkeep, control,
                     inform)
  @ccall libgalahad_single.slblt_alter_s(d::Ptr{Float32}, akeep::Ptr{Cvoid},
                                         fkeep::Ptr{Cvoid},
                                         control::Ptr{slblt_control_type{Float32,
                                                                         Int32}},
                                         inform::Ptr{slblt_inform_type{Float32,
                                                                       Int32}})::Cvoid
end

function slblt_alter(::Type{Float32}, ::Type{Int64}, d, akeep, fkeep, control,
                     inform)
  @ccall libgalahad_single_64.slblt_alter_s_64(d::Ptr{Float32},
                                               akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               control::Ptr{slblt_control_type{Float32,
                                                                               Int64}},
                                               inform::Ptr{slblt_inform_type{Float32,
                                                                             Int64}})::Cvoid
end

function slblt_alter(::Type{Float64}, ::Type{Int32}, d, akeep, fkeep, control,
                     inform)
  @ccall libgalahad_double.slblt_alter(d::Ptr{Float64}, akeep::Ptr{Cvoid},
                                       fkeep::Ptr{Cvoid},
                                       control::Ptr{slblt_control_type{Float64,
                                                                       Int32}},
                                       inform::Ptr{slblt_inform_type{Float64,
                                                                     Int32}})::Cvoid
end

function slblt_alter(::Type{Float64}, ::Type{Int64}, d, akeep, fkeep, control,
                     inform)
  @ccall libgalahad_double_64.slblt_alter_64(d::Ptr{Float64}, akeep::Ptr{Cvoid},
                                             fkeep::Ptr{Cvoid},
                                             control::Ptr{slblt_control_type{Float64,
                                                                             Int64}},
                                             inform::Ptr{slblt_inform_type{Float64,
                                                                           Int64}})::Cvoid
end

function slblt_alter(::Type{Float128}, ::Type{Int32}, d, akeep, fkeep, control,
                     inform)
  @ccall libgalahad_quadruple.slblt_alter_q(d::Ptr{Float128}, akeep::Ptr{Cvoid},
                                            fkeep::Ptr{Cvoid},
                                            control::Ptr{slblt_control_type{Float128,
                                                                            Int32}},
                                            inform::Ptr{slblt_inform_type{Float128,
                                                                          Int32}})::Cvoid
end

function slblt_alter(::Type{Float128}, ::Type{Int64}, d, akeep, fkeep, control,
                     inform)
  @ccall libgalahad_quadruple_64.slblt_alter_q_64(d::Ptr{Float128},
                                                  akeep::Ptr{Cvoid},
                                                  fkeep::Ptr{Cvoid},
                                                  control::Ptr{slblt_control_type{Float128,
                                                                                  Int64}},
                                                  inform::Ptr{slblt_inform_type{Float128,
                                                                                Int64}})::Cvoid
end
