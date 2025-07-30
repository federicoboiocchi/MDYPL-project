library("tinytest")
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

n_quad <- 100
gh <- gauss.quad(n_quad, kind = "hermite")

## Without intercept
kappa <- 0.2
gamma <- 2
alpha <- 1 / (1 + kappa)
mu <- 0.4
b <- 5.1
sigma <- 1.8

## Ensure solution from solve_se is the same as that of previous codebases
sol_fb <- MDYPL_slv(kappa, gamma, alpha, n_quad, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH", opt_met = "nleqslv")
sol_fb_vec <- MDYPL_slv(kappa, gamma, alpha, n_quad, lim_opt = 1000, start = c(0.5, 2, 1), maxi = 10000, trace = FALSE, app_met = "GH-vec", opt_met = "nleqslv")
sol_ik_mod <- solve_se(kappa, gamma, alpha, start = c(mu, b, sigma))
expect_equal(sol_fb, sol_fb_vec, tolerance = 1e-06)
expect_equal(sol_fb, sol_ik_mod, tolerance = 1e-06, check.attributes = FALSE)


## With intercept
theta0 <- 1
iota <- 1

## Equality of solutions across implementations
sol_fb <- slv_4(kappa, gamma, alpha, theta0, n_quad, lim_opt = 1000, start = c(mu, b, sigma, iota), maxit = 10000, trace = FALSE, app_met = "GH", coord_trasf = TRUE)
sol_fb_vec <- solve_se(kappa, gamma, alpha, theta0, start = c(mu, b, sigma, iota))

expect_equal(sol_fb, sol_fb_vec, tolerance = 1e-05, check.attributes = FALSE)

## se1(mu, b, sigma, 0, kappa, gamma, alpha, 0)[1:3] == se0(mu, b, sigma, kappa, gamma, alpha)
expect_equal(se1(mu, b, sigma, 0, kappa, gamma, alpha, 0)[1:3],
             se0(mu, b, sigma, kappa, gamma, alpha))
