# Federico Boiocchi 4082025

# Testing solve_mDYPL_SE over a grid of kappa, gamma, alpha (1/(1+kappa)) values
# using both TrustRegion() and NewtonRaphson()

# recorded measurements: 

# times = solution time elapsed, 
# se_parameters = estimates (mu*, b*, sigma*),
# gradients = objective function values (f(mu*),f(b*),f(sigma*)) expected to be approx. zeros
# solv = solver used (either NewtonRaphson or TrustRegion()) 


using Random, Optim, NonlinearSolve, InvertedIndices, BenchmarkTools, CSV, DataFrames
supp_path = "/Users/andre/Downloads/MDYPL_main"
include(joinpath(supp_path, "code", "methods", "mDYPL.jl"))
using .mDYPL

size = 10
k = collect(range(0.01, stop=0.95, length=size))
g = collect(range(0.5, stop=20, length=size))
a = 1.0 ./(1.0 .+k)

kappa = repeat(k,size)
gamma = repeat(g, inner=size)
alpha = repeat(a, size)

size_sq = size^2

x_init = [0.5,1.0,1.0]

function est_pars(kappa,gamma,alpha,start,solver) 
    
    x_init = start

    if solver=="Trust"
        method=TrustRegion()
    elseif solver=="Newton"
        method = NewtonRaphson()
    end

    se_parameters = Matrix{Float64}(undef, size_sq,3) 
    gradients = se_parameters
    times = Vector{Float64}(undef,size_sq)
    slv = repeat([solver],size_sq)
    
    for i in 1:size_sq 
        result = @timed solve_mDYPL_SE(kappa[i], alpha[i], gamma[i]; use_eta = false,
                      verbose = false,
                      x_init,
                      method) 
        
        times[i] = result.time

        if result.value == nothing
            se_parameters[i,:] .= NaN
            gradients[i,:] .= NaN   
        else
            se_parameters[i,:] = result.value
            out = zeros(3)
            mDYPL.mDYPL_SE!(out,result.value,kappa[i],gamma[i],alpha[i],use_eta = false)
            gradients[i,:] = out
        end
    end
mat_tot = hcat(kappa,gamma,alpha,times,se_parameters,gradients,slv)   
df_tot = DataFrame(mat_tot,[:kappa, :gamma, :alpha, :times, :mu, :b, :sigma, :f_mu, :f_b, :f_sigma, :solver])
return df_tot
end

df_trust = est_pars(kappa,gamma,alpha,start,"Trust")
df_newton = est_pars(kappa,gamma,alpha,start,"Newton")

df_test = vcat(df_trust,df_newton)
CSV.write("df_test_julia",df_test)









