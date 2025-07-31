# MDYPL project
# Author: Federico Boiocchi

# eqns_GH function that gives the value of the three
# equations of the system once given mu, b, sigma.

# arguments

# n : number of nodes for each dimension, proportional to the accuracy of the cubature,
# since we are dealing with a 2D  cubature the total number of 2D nodes is n^2.

# k : is the ratio #covariates/#observations as #obs. and p goes to +Inf

# gamma: standard deviation of the linear predictor as #obs goes to +Inf

# alpha : paramater of the DY prior

# start : a user-supplied vector (mu, sigma, b), that will be used as a starting value
# for the nleqslv algorithm.

# maxit:  for the nleqslv alg

# trace: 0, or 1 depending if we want to see a verbose output

# lim_opt: boundary of the optimization problem inside the proximal operator.

# we will assume the mode of the DY prior to be the zero vector


eqns_GH <- function(mu, b, sigma, k, gamma, alpha, n, lim_opt) {
  # libraries

  library(statmod)

  # proximal operator function

  prox <- function(x, b) {
    f <- function(u) {
      b * log(1 + exp(u)) + (1 / 2) * (x - u)^2
    }
    opt <- optimize(f, maximum = FALSE, interval = c(-lim_opt, lim_opt))
    out <- opt$minimum
    return(out)
  }

  # inverse link function

  link <- function(x) {
    exp(x) / (1 + exp(x))
  }

  # second derivative of the cumulant generating function

  link2 <- function(x) {
    exp(x) / ((1 + exp(x))^2)
  }

  # h is the function that has to be evaluated on the nodes in order to approximate
  # the three integrals via GH cubature. Depending on the value of pos (1,2,3) it will give
  # the three different integrals

  h <- function(gamma, k, alpha, mu, b, sigma, x, y, pos) {
    sd_z <- gamma
    sd_zs <- sqrt((mu^2) * (gamma^2) + k * (sigma^2))
    cov <- mu * (gamma^2)
    rho <- cov / (sd_z * sd_zs)

    B <- 2 * (1 - rho^2)
    sqrtB <- sqrt(B)

    # change of variables

    z <- x * sqrtB * gamma
    zs <- y * sqrtB * sd_zs

    C <- 1 / (2 * pi * sd_z * sd_zs * sqrt(1 - rho^2)) # normalizing constant

    jac <- B * gamma * sd_zs

    ker <- exp(-1 / B * (-2 * rho * (z / gamma) * (zs / sd_zs))) * jac # kernel of a bivariate normal without the term exp(-x^2-y^2)

    pdf <- C * ker # term exp(-x^2-y^2) not included

    # these are the 3 arguments of the expected values of the 3 non linear equations
    # multiplied by the pdf function without the term exp(-x^2-y^2). depending on the
    # pos argument we obtain different function h that have to be evaluated on the nodes
    # xi and xj

    a_frac <- (1 + alpha) / 2
    prox_val <- prox(zs + (1 + alpha) * b / 2, b)
    linkz <- link(z)
    link_prox_val <- link(prox_val)

    out <- pdf * switch(pos,
      2 * (linkz) * x * (a_frac - link_prox_val),
      2 * linkz / (1 + b * link2(prox_val)),
      2 * linkz * ((a_frac - link_prox_val)^2)
    )
    return(out)
  }

  # gaussian hermite nodes and weights

  gh <- gauss.quad(n, kind = "hermite")

  xi <- gh$nodes
  wi <- gh$weights

  # eq1 eq2 and eq3 are the three equations of the system

  eq1 <- function(gamma, k, alpha, mu, b, sigma, xi, wi) {
    int <- 0
    for (i in 1:n) {
      for (j in 1:n) {
        int <- wi[i] * wi[j] * h(gamma, k, alpha, mu, b, sigma, xi[i], xi[j], 1) + int
      }
    }
    return(int)
  }

  eq2 <- function(gamma, k, alpha, mu, b, sigma, xi, wi) {
    int <- 0
    for (i in 1:n) {
      for (j in 1:n) {
        int <- wi[i] * wi[j] * h(gamma, k, alpha, mu, b, sigma, xi[i], xi[j], 2) + int
      }
    }
    return(-1 + k + int)
  }

  eq3 <- function(gamma, k, alpha, mu, b, sigma, xi, wi) {
    int <- 0
    for (i in 1:n) {
      for (j in 1:n) {
        int <- wi[i] * wi[j] * h(gamma, k, alpha, mu, b, sigma, xi[i], xi[j], 3) + int
      }
    }
    return(-(sigma^2) + ((b^2) / (k^2)) * int)
  }

  fn <- function(par) {
    mu <- par[1]
    b <- par[2]
    sigma <- par[3]

    out <- c(
      eq1(gamma, k, alpha, mu, b, sigma, xi, wi),
      eq2(gamma, k, alpha, mu, b, sigma, xi, wi),
      eq3(gamma, k, alpha, mu, b, sigma, xi, wi)
    )
    return(out)
  }

  par <- c(mu, b, sigma)
  values <- fn(par)
  return(values)
}
