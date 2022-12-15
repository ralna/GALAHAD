using Printf

include("ugo.jl")

function main(::Type{wp}) where wp

    println("Precision: ", wp)

    # Test problem objective
    function objf(x::wp)
        a = wp(10.0)
        return x * x * cos( a*x )
    end

    # Test problem first derivative
    function gradf(x::wp)
        a = wp(10.0)
        return - a * x * x * sin( a*x ) + wp(2.0) * x * cos( a*x )
    end

    # Test problem second derivative
    function hessf(x::wp)
        a = wp(10.0)
        return - a * a* x * x * cos( a*x ) - wp(4.0) * a * x * sin( a*x ) + wp(2.0) * cos( a*x )
    end

    # Derived types
    data = Ptr{Ptr{Cvoid}}()
    control = ugo_control_type{wp}()
    inform = ugo_inform_type{wp}()

    # Initialize UGO
    status = Ref{Cint}(1)
    eval_status = Ref{Cint}()
    ugo_initialize(data, control, status)

    # Set user-defined control options
    # control.print_level = 1
    # control.maxit = 100
    # control.lipschitz_estimate_used = 3

    # Read options from specfile
    specfile = "UGO.SPC"
    ugo_read_specfile(control, specfile)

    # Test problem bounds
    x_l = wp(-1.0)
    x_u = wp(2.0)

    # Test problem objective, gradient, Hessian values
    x = wp(0.0)
    f = objf(x)
    g = gradf(x)
    h = objf(x)

    # import problem data
    ugo_import(control, data, status, x_l, x_u)

    # Set for initial entry
    status[] = 1

    # Solve the problem: min f(x), x_l <= x <= x_u
    while(true)
        # Call UGO_solve
        ugo_solve_reverse(data, status, eval_status, x, f, g, h)

        # Evaluate f(x) and its derivatives as required
        if (status >= 2)  # need objective
            f = objf(x)
            if (status >= 3)  # need first derivative
                g = gradf(x)
                if (status >= 4)  # need second derivative
                    h = hessf(x)
                end
            end
        else  # the solution has been found (or an error has occured)
            break
        end
    end

    # Record solution information
    ugo_information(data, inform, status)

    if inform.status == 0
        printf("%i evaluations. Optimal objective value = %5.2f status = %1i\n", inform.f_eval, f, inform.status)
    else
        printf("BGO_solve exit status = %1i\n", inform.status)
    end

    # Delete internal workspace
    ugo_terminate(data, control, inform)
    return 0
end

for wp in (Float32, Float64)
    main(wp)
end
