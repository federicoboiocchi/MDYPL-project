# Tests
## rm(list=ls())
# No intercept case

# Time elapsed comparison

#source("C://Users//andre//Downloads//se1.R")
#source("C://Users//andre//Downloads//solve_se.R")
#source("C://Users//andre//Downloads//se0.R")
#source("C://Users//andre//Downloads//helper-functions.R")

#library("trust")
#library("statmod")
#library("nleqslv")
#library("tictoc")
#library("viridis")

## defining the grids

library("tinytest")
library("tictoc")
library("nleqslv")
library("trust")
library("statmod")
library("ggplot2")


project_path <- "~/Repositories/MDYPL-project/"
source(file.path(project_path, "solve_se.R"))
source(file.path(project_path, "se0.R"))
source(file.path(project_path, "se1.R"))
source(file.path(project_path, "helper-functions.R"))

size <- 10
k <- seq(0.01, 0.95, length.out = size)

g <- seq(0.5, 20, length.out = size)
a <- 1 / (1 + k)
start <- c(0.5, 1, 1)

kappa <- rep(k, times = size)
gamma <- rep(g, each = size)
alpha <- rep(a, times = size)

size_sq <- size^2
times <- numeric(size_sq)


estimate_pars <- function(kappa, gamma, alpha, start, solver = "nleqslv_se", trace = 10) {
    solve_fun <- switch(solver,
                        "nleqslv_se" = nleqslv_se,
                        "trust_se" = trust_se,
                        "solve_se" = solve_se,
                        "Unknown solver")
    size_sq <- length(kappa)
    se_parameters <- gradients <- matrix(NA, size_sq, 3)
    colnames(se_parameters) <- c("mu", "b", "sigma")
    colnames(gradients) <- c("f_mu", "f_b", "f_sigma")
    for (j in 1:size_sq) {
        tic()
        out <- solve_fun(kappa[j], gamma[j], alpha[j], intercept = NULL, start = start, transform = FALSE)
        elaps <- toc(quiet = FALSE)
        gradients[j, ] <- if (solver == "trust_se") attr(out, "objective") else attr(out, "funcs")
        se_parameters[j, ] <- out
        times[j] <- as.numeric(elaps$toc-elaps$tic)
        if (isTRUE(j %% trace == 0)) {
            cat("Setting", j, "/", size_sq, "| max(|grad|) =", max(abs(gradients[j, ])), "\n")
        }
    }
    data.frame(kappa = kappa, gamma = gamma,
               alpha = alpha, elapsed = times,
               se_parameters, gradients,
               solver = solver)
}

results_nleqslv <- estimate_pars(kappa, gamma, alpha, start, solver = "nleqslv_se", trace = 1)
results_trust <- estimate_pars(kappa, gamma, alpha, start, solver = "trust_se", trace = 1)
results_solve <- estimate_pars(kappa, gamma, alpha, start, solver = "solve_se", trace = 1)

results <- rbind(results_nleqslv,
                 ## results_trust,
                 results_solve)

expect_equal(max(abs(results$f_b)), 0, tolerance = 1e-08)
expect_equal(max(abs(results$f_mu)), 0, tolerance = 1e-08)
expect_equal(max(abs(results$f_sigma)), 0, tolerance = 1e-08)

ggplot(results) +
    geom_histogram(aes(elapsed)) +
    facet_grid(~ solver) +
    theme_minimal()

ggplot(results) +
    geom_histogram(aes(elapsed)) +
    facet_grid(I(kappa > 0.5) ~ solver) +
    theme_minimal()
