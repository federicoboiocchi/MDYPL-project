using Random, Optim, NonlinearSolve, InvertedIndices, BenchmarkTools, CSV, DataFrames
supp_path = "/Users/andre/Downloads/MDYPL_main"
include(joinpath(supp_path, "code", "methods", "mDYPL.jl"))
using .mDYPL

size = 10
k = collect(range(0.01, stop=0.95, length=size))
g = collect(range(0.0, stop=20, length=size))
a = 1.0 ./(1.0 .+k)

kappa = repeat(k,size)
gamma = repeat(g, inner=size)
alpha = repeat(a, size)

size_sq = size^2

times = Vector{Float64}(undef,size_sq)

for i in 1:size_sq
    times[i] = @elapsed solve_mDYPL_SE(kappa[i], alpha[i], gamma[i]; use_eta = false,
                      verbose = false,
                      x_init = [0.5, 1.0, 1.0],
                      method = TrustRegion())  
end

times_j = DataFrame(times = times)
CSV.write("times_j.csv", times_j)







