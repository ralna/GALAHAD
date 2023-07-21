mutable struct bqp_control_type
    f_indexing::Bool
    error::Cint
    out::Cint
    print_level::Cint
    start_print::Cint
    stop_print::Cint
    print_gap::Cint
    maxit::Cint
    cold_start::Cint
    ratio_cg_vs_sd::Cint
    change_max::Cint
    cg_maxit::Cint
    sif_file_device::Cint
    infinity::Float64
    stop_p::Float64
    stop_d::Float64
    stop_c::Float64
    identical_bounds_tol::Float64
    stop_cg_relative::Float64
    stop_cg_absolute::Float64
    zero_curvature::Float64
    cpu_time_limit::Float64
    exact_arcsearch::Bool
    space_critical::Bool
    deallocate_error_fatal::Bool
    generate_sif_file::Bool
    sif_file_name::NTuple{31,Cchar}
    prefix::NTuple{31,Cchar}
    sbls_control::sbls_control_type
end

mutable struct bqp_time_type
    total::Float32
    analyse::Float32
    factorize::Float32
    solve::Float32
end

mutable struct bqp_inform_type
    status::Cint
    alloc_status::Cint
    factorization_status::Cint
    iter::Cint
    cg_iter::Cint
    obj::Float64
    norm_pg::Float64
    bad_alloc::NTuple{81,Cchar}
    time::bqp_time_type
    sbls_inform::sbls_inform_type
end

function bqp_initialize(data, control, status)
    @ccall libgalahad_double.bqp_initialize(data::Ptr{Ptr{Cvoid}},
                                            control::Ptr{bqp_control_type},
                                            status::Ptr{Cint})::Cvoid
end

function bqp_read_specfile(control, specfile)
    @ccall libgalahad_double.bqp_read_specfile(control::Ptr{bqp_control_type},
                                               specfile::Ptr{Cchar})::Cvoid
end

function bqp_import(control, data, status, n, H_type, ne, H_row, H_col, H_ptr)
    @ccall libgalahad_double.bqp_import(control::Ptr{bqp_control_type},
                                        data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint}, n::Cint,
                                        H_type::Ptr{Cchar}, ne::Cint, H_row::Ptr{Cint},
                                        H_col::Ptr{Cint}, H_ptr::Ptr{Cint})::Cvoid
end

function bqp_import_without_h(control, data, status, n)
    @ccall libgalahad_double.bqp_import_without_h(control::Ptr{bqp_control_type},
                                                  data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                                  n::Cint)::Cvoid
end

function bqp_reset_control(control, data, status)
    @ccall libgalahad_double.bqp_reset_control(control::Ptr{bqp_control_type},
                                               data::Ptr{Ptr{Cvoid}},
                                               status::Ptr{Cint})::Cvoid
end

function bqp_solve_given_h(data, status, n, h_ne, H_val, g, f, x_l, x_u, x, z, x_stat)
    @ccall libgalahad_double.bqp_solve_given_h(data::Ptr{Ptr{Cvoid}}, status::Ptr{Cint},
                                               n::Cint, h_ne::Cint, H_val::Ptr{Float64},
                                               g::Ptr{Float64}, f::Float64,
                                               x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                               x::Ptr{Float64}, z::Ptr{Float64},
                                               x_stat::Ptr{Cint})::Cvoid
end

function bqp_solve_reverse_h_prod(data, status, n, g, f, x_l, x_u, x, z, x_stat, v, prod,
                                  nz_v, nz_v_start, nz_v_end, nz_prod, nz_prod_end)
    @ccall libgalahad_double.bqp_solve_reverse_h_prod(data::Ptr{Ptr{Cvoid}},
                                                      status::Ptr{Cint}, n::Cint,
                                                      g::Ptr{Float64}, f::Float64,
                                                      x_l::Ptr{Float64},
                                                      x_u::Ptr{Float64}, x::Ptr{Float64},
                                                      z::Ptr{Float64}, x_stat::Ptr{Cint},
                                                      v::Ptr{Float64}, prod::Ptr{Float64},
                                                      nz_v::Ptr{Cint},
                                                      nz_v_start::Ptr{Cint},
                                                      nz_v_end::Ptr{Cint},
                                                      nz_prod::Ptr{Cint},
                                                      nz_prod_end::Cint)::Cvoid
end

function bqp_information(data, inform, status)
    @ccall libgalahad_double.bqp_information(data::Ptr{Ptr{Cvoid}},
                                             inform::Ptr{bqp_inform_type},
                                             status::Ptr{Cint})::Cvoid
end

function bqp_terminate(data, control, inform)
    @ccall libgalahad_double.bqp_terminate(data::Ptr{Ptr{Cvoid}},
                                           control::Ptr{bqp_control_type},
                                           inform::Ptr{bqp_inform_type})::Cvoid
end
