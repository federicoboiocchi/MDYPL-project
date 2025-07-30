library("tinytest")
library("nleqslv")
library("statmod")

source("helper-functions.R")
source("se0.R")
source("se1.R")
source("solve_se.R")


n_quad <- 50
gh <- gauss.quad(n_quad, kind = "hermite")


## Test against Julia code
library("JuliaCall")


supp_path <- "/Users/yiannis/Repositories/MDYPL/"
julia_assign("supp_path", supp_path)

## Without intercept
kappa <- 0.01
gamma <- 2
alpha <- 1 / (1 + kappa)
mu <- 0.4
b <- 5.1
sigma <- 1.8
theta0 <- 1
iota <- 1

julia_assign("kappa", kappa)
julia_assign("gamma", gamma)
julia_assign("alpha", alpha)
julia_assign("iota", iota)
julia_assign("mu", mu)
julia_assign("b", b)
julia_assign("sigma", sigma)
julia_assign("theta0", theta0)


julia_command("using Random, Optim, NonlinearSolve, InvertedIndices")
julia_command('include(joinpath(supp_path, "code", "methods", "mDYPL.jl"))')
julia_command("using .mDYPL")

## No intercept
julia_command("
  Jsol0 = solve_mDYPL_SE(kappa, alpha, gamma, use_eta = false,
                         x_init = [mu, b, sigma], verbose = true,
                         method = TrustRegion())
")

Jsol0 <- julia_eval("Jsol0")
Rsol0 <- solve_se(kappa, gamma, alpha, start = c(mu, b, sigma), transform = FALSE)
expect_equal(abs(Jsol0), abs(Rsol0), tolerance = 1e-05, check.attributes = FALSE)

## With intercept
julia_command("
  Jsol1 = solve_mDYPL_SE(kappa, alpha, gamma, theta0, use_eta = false,
                         x_init = [mu, b, sigma, iota],
                         method = TrustRegion())
")

Jsol1 <- julia_eval("Jsol1")
Rsol1 <- solve_se(kappa, gamma, alpha, theta0, start = c(0.8, 0.1, 1, 1), transform = FALSE)
expect_equal(Jsol1, Rsol1, tolerance = 1e-07, check.attributes = FALSE)


## Fix gh in test and benchmark scripts
