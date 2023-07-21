mutable struct presolve_control_type
    f_indexing::Bool
    termination::Cint
    max_nbr_transforms::Cint
    max_nbr_passes::Cint
    c_accuracy::Float64
    z_accuracy::Float64
    infinity::Float64
    out::Cint
    errout::Cint
    print_level::Cint
    dual_transformations::Bool
    redundant_xc::Bool
    primal_constraints_freq::Cint
    dual_constraints_freq::Cint
    singleton_columns_freq::Cint
    doubleton_columns_freq::Cint
    unc_variables_freq::Cint
    dependent_variables_freq::Cint
    sparsify_rows_freq::Cint
    max_fill::Cint
    transf_file_nbr::Cint
    transf_buffer_size::Cint
    transf_file_status::Cint
    transf_file_name::NTuple{31,Cchar}
    y_sign::Cint
    inactive_y::Cint
    z_sign::Cint
    inactive_z::Cint
    final_x_bounds::Cint
    final_z_bounds::Cint
    final_c_bounds::Cint
    final_y_bounds::Cint
    check_primal_feasibility::Cint
    check_dual_feasibility::Cint
    pivot_tol::Float64
    min_rel_improve::Float64
    max_growth_factor::Float64
end

mutable struct presolve_inform_type
    status::Cint
    status_continue::Cint
    status_continued::Cint
    nbr_transforms::Cint
    message::NTuple{3,NTuple{81,Cchar}}
end

function presolve_initialize(data, control, status)
    @ccall libgalahad_double.presolve_initialize(data::Ptr{Ptr{Cvoid}},
                                                 control::Ref{presolve_control_type},
                                                 status::Ptr{Cint})::Cvoid
end

function presolve_read_specfile(control, specfile)
    @ccall libgalahad_double.presolve_read_specfile(control::Ref{presolve_control_type},
                                                    specfile::Ptr{Cchar})::Cvoid
end

function presolve_import_problem(control, data, status, n, m, H_type, H_ne, H_row, H_col,
                                 H_ptr, H_val, g, f, A_type, A_ne, A_row, A_col, A_ptr,
                                 A_val, c_l, c_u, x_l, x_u, n_out, m_out, H_ne_out,
                                 A_ne_out)
    @ccall libgalahad_double.presolve_import_problem(control::Ref{presolve_control_type},
                                                     data::Ptr{Ptr{Cvoid}},
                                                     status::Ptr{Cint}, n::Cint, m::Cint,
                                                     H_type::Ptr{Cchar}, H_ne::Cint,
                                                     H_row::Ptr{Cint}, H_col::Ptr{Cint},
                                                     H_ptr::Ptr{Cint}, H_val::Ptr{Float64},
                                                     g::Ptr{Float64}, f::Float64,
                                                     A_type::Ptr{Cchar}, A_ne::Cint,
                                                     A_row::Ptr{Cint}, A_col::Ptr{Cint},
                                                     A_ptr::Ptr{Cint}, A_val::Ptr{Float64},
                                                     c_l::Ptr{Float64}, c_u::Ptr{Float64},
                                                     x_l::Ptr{Float64}, x_u::Ptr{Float64},
                                                     n_out::Ptr{Cint}, m_out::Ptr{Cint},
                                                     H_ne_out::Ptr{Cint},
                                                     A_ne_out::Ptr{Cint})::Cvoid
end

function presolve_transform_problem(data, status, n, m, H_ne, H_col, H_ptr, H_val, g, f,
                                    A_ne, A_col, A_ptr, A_val, c_l, c_u, x_l, x_u, y_l, y_u,
                                    z_l, z_u)
    @ccall libgalahad_double.presolve_transform_problem(data::Ptr{Ptr{Cvoid}},
                                                        status::Ptr{Cint}, n::Cint, m::Cint,
                                                        H_ne::Cint, H_col::Ptr{Cint},
                                                        H_ptr::Ptr{Cint},
                                                        H_val::Ptr{Float64},
                                                        g::Ptr{Float64}, f::Ptr{Float64},
                                                        A_ne::Cint, A_col::Ptr{Cint},
                                                        A_ptr::Ptr{Cint},
                                                        A_val::Ptr{Float64},
                                                        c_l::Ptr{Float64},
                                                        c_u::Ptr{Float64},
                                                        x_l::Ptr{Float64},
                                                        x_u::Ptr{Float64},
                                                        y_l::Ptr{Float64},
                                                        y_u::Ptr{Float64},
                                                        z_l::Ptr{Float64},
                                                        z_u::Ptr{Float64})::Cvoid
end

function presolve_restore_solution(data, status, n_in, m_in, x_in, c_in, y_in, z_in, n, m,
                                   x, c, y, z)
    @ccall libgalahad_double.presolve_restore_solution(data::Ptr{Ptr{Cvoid}},
                                                       status::Ptr{Cint}, n_in::Cint,
                                                       m_in::Cint, x_in::Ptr{Float64},
                                                       c_in::Ptr{Float64},
                                                       y_in::Ptr{Float64},
                                                       z_in::Ptr{Float64}, n::Cint,
                                                       m::Cint, x::Ptr{Float64},
                                                       c::Ptr{Float64}, y::Ptr{Float64},
                                                       z::Ptr{Float64})::Cvoid
end

function presolve_information(data, inform, status)
    @ccall libgalahad_double.presolve_information(data::Ptr{Ptr{Cvoid}},
                                                  inform::Ref{presolve_inform_type},
                                                  status::Ptr{Cint})::Cvoid
end

function presolve_terminate(data, control, inform)
    @ccall libgalahad_double.presolve_terminate(data::Ptr{Ptr{Cvoid}},
                                                control::Ref{presolve_control_type},
                                                inform::Ref{presolve_inform_type})::Cvoid
end
