# Federico Boiocchi 482025

### Analysis

# R and Julia solvers Comparison
library(viridis)
library(ggplot2)
library(tinytest)

# files required:

# df_test_R.csv from R_test_solver.R 
# df_test_julia.csv from Julia_test_solver.jl

# Same input arguments of the R and Julia Test scripts

# (remark: Size has to be increased for a more detailed analysis)

# Input Arguments
size <- 5  
k <- seq(0.01, 0.95, length.out = size)

g <- seq(0.5, 20, length.out = size)
a <- 1 / (1 + k)
start <- c(0.5, 1, 1)

# grid of kappa,gamma,alpha 

kappa <- rep(k, times = size)
gamma <- rep(g, each = size)
alpha <- rep(a, times = size) # not free to vary

size_sq <- size^2

# R Analysis

df_R <- read.csv("df_test_R")

## Elapsed time

contour_plot <- function(z, method, what, lang) {
  filled.contour(k, g, z,
    color.palette = viridis,
    xlab = "kappa", ylab = "gamma",
    main = paste0(what, sep = " ", lang), cex.main = 0.9
  )
  mtext(paste0("solver: ", method), side = 3, line = 0.45, cex = 0.8, adj = 0.27)
}

t_nleqslv_R <- df_R[df_R$solver == "nleqslv_se", ]$elapsed
t_solve_R <- df_R[df_R$solver == "solve_se", ]$elapsed
t_trust_R <- df_R[df_R$solver == "trust_se", ]$elapsed

t_nleqslv_R <- matrix(data = t_nleqslv_R, ncol = size, nrow = size)
t_solve_R <- matrix(data = t_solve_R, ncol = size, nrow = size)
t_trust_R <- matrix(data = t_trust_R, ncol = size, nrow = size)

contour_plot(t_nleqslv_R, "nleqslv_se", "Elapsed Time (seconds)", "R")
contour_plot(t_solve_R, "solve_se", "Elapsed Time (seconds)", "R")
contour_plot(t_trust_R, "trust_se", "Elapsed Time (seconds)", "R")

# Convergence always reached
expect_equal(max(abs(df_R$f_b)), 0, tolerance = 1e-08)
expect_equal(max(abs(df_R$f_mu)), 0, tolerance = 1e-08)
expect_equal(max(abs(df_R$f_sigma)), 0, tolerance = 1e-08)

# Accuracy of the solution measured through gradients
# solver: nleqslv_se

df_R_nleqslv <- df_R[df_R$solver == "nleqslv_se", ]

f_b_nleqslv <- matrix(df_R_nleqslv$f_b, size, size)
f_mu_nleqslv <- matrix(df_R_nleqslv$f_mu, size, size)
f_sigma_nleqslv <- matrix(df_R_nleqslv$f_sigma, size, size)

contour_plot(f_b_nleqslv, "nleqslv_se", "Gradient: f_b", "R")
contour_plot(f_mu_nleqslv, "nleqslv_se", "Gradient: f_mu", "R")
contour_plot(f_sigma_nleqslv, "nleqslv_se", "Gradient: f_sigma", "R")

# Distribution of times over two methods

# Distribution over the entire set of combination of kappa,gamma,alpha

ggplot(df_R) +
  geom_histogram(aes(elapsed)) +
  facet_grid(~solver) +
  theme_minimal()

# Distribution over a subset of the combinations of kappa, gamma, alpha
# k > 0.5

ggplot(df_R) +
  geom_histogram(aes(elapsed)) +
  facet_grid(I(kappa > 0.5) ~ solver) +
  theme_minimal()


# JULIA Analysis

df_j <- read.csv("df_test_julia")

# Testing convergence
expect_equal(max(abs(df_j$f_b)), 0, tolerance = 1e-08)
expect_equal(max(abs(df_j$f_mu)), 0, tolerance = 1e-08)
expect_equal(max(abs(df_j$f_sigma)), 0, tolerance = 1e-08)

# The convergence test is not passed

# investigating the non-convergence settings, by isolating indices
# associated with combination of (kappa,gamma) leading to non convergence

conv_lim <- 1e-6

not_conv_ind_b <- which(df_j$f_b > conv_lim)
not_conv_ind_mu <- which(df_j$f_mu > conv_lim)
not_conv_ind_sigma <- which(df_j$f_sigma > conv_lim)

# indexes for which at least one gradient is approximately different from 0

nc_ind <- unique(c(not_conv_ind_b, not_conv_ind_mu, not_conv_ind_sigma))
nc_ind_trust <- nc_ind[nc_ind %in% c(1:size_sq)]
nc_ind_newton <- nc_ind[nc_ind %in% c(size_sq:c(2 * size_sq))]

# Indices of combinations of kappa and gamma associated with non convergence
# splitted across solver method (trust or newton)

kappa_nc_t <- df_j$kappa[nc_ind_trust]
gamma_nc_t <- df_j$gamma[nc_ind_trust]

kappa_nc_nr <- df_j$kappa[nc_ind_newton]
gamma_nc_nr <- df_j$gamma[nc_ind_newton]

## Elapsed_time

t_trust_j <- df_j[df_j$solver == "Trust", ]$elapsed
t_newton_j <- df_j[df_j$solver == "Newton", ]$elapsed

t_trust_j <- matrix(data = t_trust_j, ncol = size, nrow = size)
t_newton_j <- matrix(data = t_newton_j, ncol = size, nrow = size)

# Contour plots with elapsed times for TrustRegion()
# and NewtonRaphson()

# red dots identify combination of (kappa,gamma)
# associated with non convergence

contour_plot(t_trust_j, "TrustRegion()","Elapsed time (seconds)","Julia")
points(x = kappa_nc_t, y = gamma_nc_t, pch = 18, col = "red")

contour_plot(t_newton_j, "NewtonRaphson()","Elapsed time (seconds)","Julia")
points(x = kappa_nc_nr, y = gamma_nc_nr, pch = 19, col = "red")

# Accuracy of the solution measured through gradients
# solver: NewtonRaphson()

df_j_nr <- df_j[df_j$solver == "Newton", ]

f_b_nr <- matrix(df_j_nr$f_b, size, size)
f_mu_nr <- matrix(df_j_nr$f_mu, size, size)
f_sigma_nr <- matrix(df_j_nr$f_sigma, size, size)

contour_plot(f_b_nr, "NewtonRaphson()", "Gradient: f_b", "Julia")
contour_plot(f_mu_nr, "NewtonRaphson()", "Gradient: f_mu", "Julia")
contour_plot(f_sigma_nr, "NewtonRaphson()", "Gradient: f_sigma", "Julia")

