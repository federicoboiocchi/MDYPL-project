# Federico Boiocchi (782025)

# test-solve_se.R

project_path <- "C:/Users/andre/Documents/brglm2/R"
#project_path <- "~/brglm2/R"

source(file.path(project_path, "solve_se.R"))
source(file.path(project_path, "se.R"))
source(file.path(project_path, "utils.R"))

library("statmod")
library("nleqslv")
library("tictoc")
library("tinytest")
library("viridis")
library("ggplot2")

# Input arguments

size <- 25
k <- seq(0.01, 0.95, length.out = size)

g <- seq(0.5, 20, length.out = size)
a <- 1 / (1 + k)
start <- c(0.5, 1, 1)

kappa <- rep(k, times = size)
gamma <- rep(g, each = size)
alpha <- rep(a, times = size)

size_sq <- size^2
times <- numeric(size_sq)
init_iter <- 5

# non linear system solved via nleqslv()

# first iterations carried out with optim() using Nelder-Mead method

estimate_pars <- function(kappa, gamma, alpha, start, trace = 10) {
  
  size_sq <- length(kappa)
  se_parameters <- gradients <- matrix(NA, size_sq, 3)
  colnames(se_parameters) <- c("mu", "b", "sigma")
  colnames(gradients) <- c("f_mu", "f_b", "f_sigma")
  for (j in 1:size_sq) {
    tic()
    out <- solve_se(kappa = kappa[j], ss = gamma[j], alpha[j], intercept = NULL, start = start,
                     ,corrupted=FALSE,transform = FALSE,init_method = "Nelder-Mead",init_iter = init_iter)
    elaps <- toc(quiet = FALSE)
    gradients[j, ] <- attr(out, "funcs")
    se_parameters[j, ] <- out[1:3]
    times[j] <- as.numeric(elaps$toc-elaps$tic)
    if (isTRUE(j %% trace == 0)) {
      cat("Setting", j, "/", size_sq, "| max(|grad|) =", max(abs(gradients[j, ])), "\n")
    }
  }
  data.frame(kappa = kappa, gamma = gamma,
             alpha = alpha, elapsed = times,
             se_parameters, gradients)
}

results_opt <- estimate_pars(kappa, gamma, alpha, start, trace = 1)

# write.csv(results_opt, "df_test_R_opt")

# dfR <- read.csv("df_test_R_opt")

dfR <- results_opt

dim(dfR)

## Elapsed time

tR <- dfR$elapsed
tR <- matrix(tR,size,size)

contour_plot <- function(k,g,z, method, what) {
    filled.contour(k, g, z,
                   color.palette = viridis,
                   xlab = "kappa", ylab = "gamma",
                   main = what, cex.main = 0.9
    )
    mtext(paste0("solver: ", method), side = 3, line = 0.45, cex = 0.8, adj = 0.27)
}

contour_plot(k,g,tR, "nleqslv() and optim()", "Elapsed Time (seconds) R")

# Investigate the area where the elapsed time is higher

# Convergence check

expect_equal(max(abs(results_opt$f_b)), 0, tolerance = 1e-08)
expect_equal(max(abs(results_opt$f_mu)), 0, tolerance = 1e-07) # not passed at 1e-08
expect_equal(max(abs(results_opt$f_sigma)), 0, tolerance = 1e-08)


# Accuracy of the solution measured through gradients
# solver: first init_iter with optim() then nleqslv()

f_b <- matrix(dfR$f_b, size, size)
f_mu <- matrix(dfR$f_mu, size, size)
f_sigma <- matrix(dfR$f_sigma, size, size)

contour_plot(k,g,f_b, "nleqslv_se and optim()", "Gradient: f_b")
contour_plot(k,g,f_mu, "nleqslv_se and optim()", "Gradient: f_mu")
contour_plot(k,g,f_sigma, "nleqslv_se and optim()", "Gradient: f_sigma")

# Distribution of times over two methods

# Distribution over the entire set of combination of kappa,gamma,alpha

ggplot(dfR) +
  geom_histogram(aes(elapsed)) +
  theme_minimal()

# Distribution of elapsed times over a subset of the combinations of kappa, gamma, alpha
# k > 0.5

ggplot(dfR) +
  geom_histogram(aes(elapsed)) +
  facet_grid(I(kappa > 0.5)) +
  theme_minimal()

ggplot(dfR) +
  geom_histogram(aes(elapsed)) +
  facet_grid(I(gamma > 12)) +
  theme_minimal()

# remark:
# we obtain similar distribution of elapsed time by splitting the gamma 
# axis (it seems that the distribution of elapsed times is invariant of gamma). 
# On the contrary we obtain different distribution if we split the 
# k axis; for low k the distr. is more concentrated towards low times while
# for high k is more widespread. 

# Values for mu*,b*,sigma*, across all parameters combinations

mu_grid <- matrix(dfR$mu,size,size)
b_grid <- matrix(dfR$b,size,size)
sigma_grid <- matrix(dfR$sigma,size,size)

contour_plot(k,g,mu_grid, "nleqslv_se and optim()", expression(mu["*"]))
contour_plot(k,g,b_grid, "nleqslv_se and optim()", expression(b["*"]))
contour_plot(k,g,sigma_grid, "nleqslv_se and optim()", expression(sigma["*"]))

# aggregate MSE (aMSE)

# aMSE = sigma*^2+(1-mu*)^2*((gamma)^2)/k
mu <- dfR$mu;b <- dfR$b;sigma<-dfR$sigma
aMSE <- (sigma)^2+((1-mu)^2)*(((gamma)^2)/kappa)
rMSE <- matrix(sqrt(aMSE),size,size)
contour_plot(k,g,rMSE, "nleqslv_se and optim()",expression("root aMSE"==sigma["*"]^2+(1-mu["*"])^2*gamma^2/k))


