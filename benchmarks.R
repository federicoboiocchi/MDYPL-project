library("microbenchmark")
library("nleqslv")
library("statmod")
library("cubature")

source("helper-functions.R")
source("se0.R")
source("se1.R")
source("solve_se.R")
source("backup/eqns_GH.R")
source("backup/eqns_GH_vec.R")
source("backup/MDYPL_slv.R")
source("backup/SE_with_intercept/eq_cub4.R")
source("backup/SE_with_intercept/eqns4.R")
source("backup/SE_with_intercept/slv_4.R")

n_quad <- 50
gh <- gauss.quad(n_quad, kind = "hermite")

## Without intercept
kappa <- 0.2
gamma <- 2
alpha <- 1 / (1 + kappa)
mu <- 0.4
b <- 5.1
sigma <- 1.8

## Benchmark function evaluations
microbenchmark(
    fb = eqns_GH(mu, b, sigma, kappa, gamma, alpha, n = n_quad, lim_opt = 100),
    fb_vec = eqns_GH_vec(mu, b, sigma, kappa, gamma, alpha, n = n_quad, lim_opt = 100),
    ik_mod = se0(mu, b, sigma, kappa, gamma, alpha, gh = gh),
times = 50)

## Benchmark optimization
microbenchmark(
    fb = MDYPL_slv(kappa, gamma, alpha, n_quad, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH", opt_met = "nleqslv"),
    fb_vec = MDYPL_slv(kappa, gamma, alpha, n_quad, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH-vec", opt_met = "nleqslv"),
    ik_mod = solve_se(kappa, gamma, alpha, start = c(mu, b, sigma), gh = gh),
    times = 10)



## With intercept
theta0 <- 1
iota <- 1

## Benchmark function evaluations
microbenchmark(
    fb = eqns4(gamma, kappa, alpha, mu, b, sigma, iota, theta0, lim_opt = 100, n = n_quad, method = "GH", coord_trasf = TRUE),
    fb_cub = eq_cub4(gamma, kappa, alpha, mu, b, sigma, iota, theta0, lim_opt = 100, coord_trasf = TRUE),
    fb_vec = se1(mu, b, sigma, iota, kappa, gamma, alpha, theta0, gh = gh),
    times = 10)


## Benchmark optimization
microbenchmark(
    fb = slv_4(kappa, gamma, alpha, theta0, n_quad, lim_opt = 1000, start = c(mu, b, sigma, iota), maxit = 10000, trace = FALSE, app_met = "GH", coord_trasf = TRUE),
    fb_vec = solve_se(kappa, gamma, alpha, theta0, start = c(mu, b, sigma, iota), gh = gh),
    times = 10)
