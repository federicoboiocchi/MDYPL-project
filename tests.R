library("nleqslv")
library("statmod")

library("tinytest")
library("microbenchmark")


source("eqns_GH.R")
source("eqns_GH_vec.R")
source("eqns_GH2.R")
source("MDYPL_slv.R")

kappa <- 0.2
gamma <- 2
alpha <- 1 / (1 + kappa)

mu <- 0.4
b <- 5.1
sigma <- 1.8

gh <- gauss.quad(50, kind = "hermite")

## Benchmark function evaluations
b_fun <- microbenchmark(
    fb = eqns_GH(mu, b, sigma, kappa, gamma, alpha, n = 50, lim_opt = 100),
    fb_vec = eqns_GH_vec(mu, b, sigma, kappa, gamma, alpha, n = 50, lim_opt = 100),
    ik_mod = eqns_GH2(mu, b, sigma, kappa, gamma, alpha, gh = gh),
times = 50)

## Benchmark optimization
microbenchmark(
    fb = MDYPL_slv(kappa, gamma, alpha, 50, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH", opt_met = "nleqslv"),
    fb_vec = MDYPL_slv(kappa, gamma, alpha, 50, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH-vec", opt_met = "nleqslv"),
    ik_mod = solve_mDYPL_SE(kappa, gamma, alpha, start = c(mu, b, sigma)),
times = 10)

## Check equality of solutions
sol_fb <- MDYPL_slv(kappa, gamma, alpha, 50, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH", opt_met = "nleqslv")
sol_fb_vec <- MDYPL_slv(kappa, gamma, alpha, 50, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH-vec", opt_met = "nleqslv")
sol_ik_mod <- solve_mDYPL_SE(kappa, gamma, alpha, start = c(mu, b, sigma))

expect_equal(sol_fb, sol_fb_vec, tolerance = 1e-06)
expect_equal(sol_fb, sol_ik_mod, tolerance = 1e-06, check.attributes = FALSE)
