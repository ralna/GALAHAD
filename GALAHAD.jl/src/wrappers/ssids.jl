mutable struct spral_ssids_options
    array_base::Cint
    print_level::Cint
    unit_diagnostics::Cint
    unit_error::Cint
    unit_warning::Cint
    ordering::Cint
    nemin::Cint
    ignore_numa::Bool
    use_gpu::Bool
    gpu_only::Bool
    min_gpu_work::Int64
    max_load_inbalance::Cfloat
    gpu_perf_coeff::Cfloat
    scaling::Cint
    small_subtree_threshold::Int64
    cpu_block_size::Cint
    action::Bool
    pivot_method::Cint
    small::Float64
    u::Float64
    nstream::Cint
    multiplier::Float64
    min_loadbalance::Cfloat
    failed_pivot_method::Cint
end

mutable struct spral_ssids_inform
    flag::Cint
    matrix_dup::Cint
    matrix_missing_diag::Cint
    matrix_outrange::Cint
    matrix_rank::Cint
    maxdepth::Cint
    maxfront::Cint
    num_delay::Cint
    num_factor::Int64
    num_flops::Int64
    num_neg::Cint
    num_sup::Cint
    num_two::Cint
    stat::Cint
    cuda_error::Cint
    cublas_error::Cint
    not_first_pass::Cint
    not_second_pass::Cint
    nparts::Cint
    cpu_flops::Int64
    gpu_flops::Int64
end

function spral_ssids_default_options(options)
    @ccall libgalahad_double.spral_ssids_default_options(options::Ptr{spral_ssids_options})::Cvoid
end

function spral_ssids_analyse(check, n, order, ptr, row, val, akeep, options, inform)
    @ccall libgalahad_double.spral_ssids_analyse(check::Bool, n::Cint, order::Ptr{Cint},
                                                 ptr::Ptr{Int64}, row::Ptr{Cint},
                                                 val::Ptr{Float64}, akeep::Ptr{Ptr{Cvoid}},
                                                 options::Ptr{spral_ssids_options},
                                                 inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_analyse_ptr32(check, n, order, ptr, row, val, akeep, options, inform)
    @ccall libgalahad_double.spral_ssids_analyse_ptr32(check::Bool, n::Cint,
                                                       order::Ptr{Cint}, ptr::Ptr{Cint},
                                                       row::Ptr{Cint}, val::Ptr{Float64},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       options::Ptr{spral_ssids_options},
                                                       inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_analyse_coord(n, order, ne, row, col, val, akeep, options, inform)
    @ccall libgalahad_double.spral_ssids_analyse_coord(n::Cint, order::Ptr{Cint}, ne::Int64,
                                                       row::Ptr{Cint}, col::Ptr{Cint},
                                                       val::Ptr{Float64},
                                                       akeep::Ptr{Ptr{Cvoid}},
                                                       options::Ptr{spral_ssids_options},
                                                       inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_factor(posdef, ptr, row, val, scale, akeep, fkeep, options, inform)
    @ccall libgalahad_double.spral_ssids_factor(posdef::Bool, ptr::Ptr{Int64},
                                                row::Ptr{Cint}, val::Ptr{Float64},
                                                scale::Ptr{Float64}, akeep::Ptr{Cvoid},
                                                fkeep::Ptr{Ptr{Cvoid}},
                                                options::Ptr{spral_ssids_options},
                                                inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_factor_ptr32(posdef, ptr, row, val, scale, akeep, fkeep, options,
                                  inform)
    @ccall libgalahad_double.spral_ssids_factor_ptr32(posdef::Bool, ptr::Ptr{Cint},
                                                      row::Ptr{Cint}, val::Ptr{Float64},
                                                      scale::Ptr{Float64},
                                                      akeep::Ptr{Cvoid},
                                                      fkeep::Ptr{Ptr{Cvoid}},
                                                      options::Ptr{spral_ssids_options},
                                                      inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_solve1(job, x1, akeep, fkeep, options, inform)
    @ccall libgalahad_double.spral_ssids_solve1(job::Cint, x1::Ptr{Float64},
                                                akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                options::Ptr{spral_ssids_options},
                                                inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_solve(job, nrhs, x, ldx, akeep, fkeep, options, inform)
    @ccall libgalahad_double.spral_ssids_solve(job::Cint, nrhs::Cint, x::Ptr{Float64},
                                               ldx::Cint, akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               options::Ptr{spral_ssids_options},
                                               inform::Ptr{spral_ssids_inform})::Cvoid
end

function spral_ssids_free_akeep(akeep)
    @ccall libgalahad_double.spral_ssids_free_akeep(akeep::Ptr{Ptr{Cvoid}})::Cint
end

function spral_ssids_free_fkeep(fkeep)
    @ccall libgalahad_double.spral_ssids_free_fkeep(fkeep::Ptr{Ptr{Cvoid}})::Cint
end

function spral_ssids_free(akeep, fkeep)
    @ccall libgalahad_double.spral_ssids_free(akeep::Ptr{Ptr{Cvoid}},
                                              fkeep::Ptr{Ptr{Cvoid}})::Cint
end

function spral_ssids_enquire_posdef(akeep, fkeep, options, inform, d)
    @ccall libgalahad_double.spral_ssids_enquire_posdef(akeep::Ptr{Cvoid},
                                                        fkeep::Ptr{Cvoid},
                                                        options::Ptr{spral_ssids_options},
                                                        inform::Ptr{spral_ssids_inform},
                                                        d::Ptr{Float64})::Cvoid
end

function spral_ssids_enquire_indef(akeep, fkeep, options, inform, piv_order, d)
    @ccall libgalahad_double.spral_ssids_enquire_indef(akeep::Ptr{Cvoid}, fkeep::Ptr{Cvoid},
                                                       options::Ptr{spral_ssids_options},
                                                       inform::Ptr{spral_ssids_inform},
                                                       piv_order::Ptr{Cint},
                                                       d::Ptr{Float64})::Cvoid
end

function spral_ssids_alter(d, akeep, fkeep, options, inform)
    @ccall libgalahad_double.spral_ssids_alter(d::Ptr{Float64}, akeep::Ptr{Cvoid},
                                               fkeep::Ptr{Cvoid},
                                               options::Ptr{spral_ssids_options},
                                               inform::Ptr{spral_ssids_inform})::Cvoid
end
