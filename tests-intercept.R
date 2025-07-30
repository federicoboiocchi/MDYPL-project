library("nleqslv")
library("statmod")
library("cubature")

library("tinytest")
library("microbenchmark")

source("SE-no-intercept.R")
source("SE-intercept.R")
source("backup/SE_with_intercept/eq_cub4.R")
source("backup/SE_with_intercept/eqns4.R")
source("backup/SE_with_intercept/slv_4.R")

kappa <- 0.2
gamma <- 2
alpha <- 1 / (1 + kappa)
theta0 <- 1

mu <- 0.4
b <- 5.1
sigma <- 1.8
iota <- 2

gh <- gauss.quad(50, kind = "hermite")

## Benchmark function evaluations
microbenchmark(
    fb = eqns4(gamma, kappa, alpha, mu, b, sigma, iota, theta0, lim_opt = 100, n = 50, method = "GH", coord_trasf = TRUE),
    fb_cub = eq_cub4(gamma, kappa, alpha, mu, b, sigma, iota, theta0, lim_opt = 100, coord_trasf = TRUE),
    fb_vec = mdypl_se4(mu, b, sigma, iota, kappa, gamma, alpha, theta0, gh = gh),
times = 50)

## Benchmark optimization
microbenchmark(
    fb = slv_4(kappa, gamma, alpha, theta0, 50, lim_opt = 1000, start = c(0.5, 2, 1, 1), maxit = 10000, trace = FALSE, app_met = "GH", coord_trasf = TRUE),
    fb_vec = solve_mdypl_se4(kappa, gamma, alpha, theta0, start = c(mu, b, sigma, iota)),
times = 10)

## Check equality of solutions
sol_fb <- slv_4(kappa, gamma, alpha, theta0, 50, lim_opt = 1000, start = c(0.5, 2, 1, 1), maxit = 10000, trace = FALSE, app_met = "GH", coord_trasf = TRUE)
sol_fb_vec <- solve_mdypl_se4(kappa, gamma, alpha, theta0, start = c(mu, b, sigma, iota))

expect_equal(sol_fb, sol_fb_vec, tolerance = 1e-05, check.attributes = FALSE)

