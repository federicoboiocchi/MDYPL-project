# first equation non linear system with intercept

rm(list = ls())
library(statmod)


n <- 150 
out <- gauss.quad(n, kind = "hermite")

x <- out$nodes
w <- out$weights

mu <- 2
b <- 1
sigma <- 3
iota <- 1.7
t0 <- 1.1

lim_opt <- 1250

gamma <- 4
k <- 0.3
alpha <- 1 / (1 + k)


eq1_GH <- function(gamma, k, alpha, mu, b, sigma, iota, t0, x, w, n, lim_opt) {

  mu_z1 <- t0
  mu_z2 <- iota

  var_z1 <- (mu * gamma)^2
  var_z2 <- ((mu * gamma)^2) + k * sigma^2

  sd_z1 <- sqrt(var_z1)
  sd_z2 <- sqrt(var_z2)

  cov_z1z2 <- (mu * gamma)^2

  rho_z1z2 <- cov_z1z2 / (sd_z1 * sd_z2)

  B <- sqrt(2 * (1 - rho_z1z2^2))

  jacs_prod <- mu * gamma * (B^2) * sd_z2

  link <- function(x) {
    1 / (1 + exp(-x))
  }

  prox <- function(x, b) {
    f <- function(u) {
      b * log(1 + exp(u)) + (1 / 2) * (x - u)^2
    }
    opt <- optimize(f, maximum = FALSE, interval = c(-lim_opt, lim_opt))
    out <- opt$minimum
    return(out)
  }

  Q <- function(alpha, b, z) {
    (1/(1 + alpha)) - link(prox(z + (b/(1 + alpha)), b))
  }

  g_func <- function(z1, z2) {
    link(z1) * z1 * Q(alpha, b, z2) - link(-z1) * z1 * Q(alpha, b, -z2)
  }

  C <- 1 / (2 * pi * sd_z1 * sd_z2 * sqrt(1 - rho_z1z2^2))

  h <- function(x, y) {
    z <- x * B
    g <- B * (y * sd_z2 - (mu * gamma) * x) / (sqrt(k) * sigma)

    z1 <- mu * gamma * z + t0
    z2 <- mu * gamma * z + sqrt(k) * sigma * g + iota

    out <- g_func(z1, z2) * jacs_prod * C * exp((-1 / (B^2)) * (-2 * rho_z1z2) * (z) * ((mu * gamma * z + sqrt(k) * sigma * g) / (sd_z2)))
    return(out)
  }

  int <- 0
  for (i in 1:n) {
    for (j in 1:n) {
      int <- w[i] * w[j] * h(x[i], x[j]) + int
    }
  }
  return(int)
}


eq1_GH(gamma, k, alpha, mu, b, sigma, iota, t0, x, w, n, lim_opt) 