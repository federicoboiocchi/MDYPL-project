# MDYPL project
# Author: Federico Boiocchi

# eqns_GH_vec is a function from R^3 to R^3. For a specific vector (mu,b,sigma) it gives
# the values of the three components of the nonlinear system that has to be solved in order to find
# (mu*,b*,sigma*) which are necessary to described the behavior of beta_hat_DY when p/n->k in (0,1).

# this function is vectorized in the computation of the GH approximation
# avoiding (3) double nested for cycle.

eqns_GH_vec <- function(mu, b, sigma, k, gamma, alpha, n, lim_opt) {
  # library

  library(statmod) # used to generate GH nodes

  # proximal operator function

  prox <- function(x, b) {
    f <- function(u) {
      b * log(1 + exp(u)) + (1 / 2) * (x - u)^2
    }
    opt <- optimize(f, maximum = FALSE, interval = c(-lim_opt, lim_opt))
    out <- opt$minimum
    return(out)
  }

  prox <- Vectorize(prox, vectorize.args = "x")

  # inverse link function (first derivative of the cumulant generating function)
  # inverse logit

  link <- function(x) {
    exp(x) / (1 + exp(x))
  }

  # second derivative of the cumulant generating function

  link2 <- function(x) {
    exp(x) / ((1 + exp(x))^2)
  }

  # h is the function from R^3 to R that has to be evaluated on the GH nodes in order to approximate
  # each one of the three expected values (integrals) via Gaussian-Hermite cubature. Depending on the value of pos (1,2,3) it will give
  # the three different equations of the non linear system

  # gamma: limit of the sd of the linear predictor
  # k: ratio between #explanatory variables and sample size
  # alpha: hyperparameter of the DY prior
  # mu: scaling factor to adjust beta_hat_DY
  # b: parameter to adjust the LRT statistics
  # sigma: variance used to adjust tha asymptotic behavior of beta_hat_DY
  # x,y: vector of nodes on which h has to be aevaluated, h has to be evaluated on expand.grid(x,y)
  # pos: argument to choose which transformation has to be used to compute the expeted value


  h <- function(gamma, k, alpha, mu, b, sigma, x, y, pos) {
    sd_z <- gamma # sd of Z
    sd_zs <- sqrt((mu^2) * (gamma^2) + k * (sigma^2)) # sd of Z* (Z star)
    cov <- mu * (gamma^2) # covariance between Z and Z*
    rho <- cov / (sd_z * sd_zs) # correlation coeff. between Z and Z*

    B <- 2 * (1 - rho^2)
    sqrtB <- sqrt(B)

    # change of variables in order to recover an integrand of the form h(x,y)*exp(-x^2-y^2)

    z <- x * sqrtB * gamma
    zs <- y * sqrtB * sd_zs

    C <- 1 / (2 * pi * sd_z * sd_zs * sqrt(1 - rho^2)) # normalizing constant

    jac <- B * gamma * sd_zs # jacobian

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
      2 * (linkz) * z * (a_frac - link_prox_val),
      2 * linkz / (1 + b * link2(prox_val)),
      2 * linkz * ((a_frac - link_prox_val))^2
    )
    return(out)
  }

  # Gaussian Hermite nodes and weights

  gh <- gauss.quad(n, kind = "hermite")

  xi <- gh$nodes
  wi <- matrix(data = gh$weights, nrow = n, ncol = 1)

  # eq1 eq2 and eq3 are the three equations of the system

  eq1 <- function(gamma, k, alpha, mu, b, sigma, xi, wi) {
    h1 <- function(x, y) {
      out <- h(gamma, k, alpha, mu, b, sigma, x, y, 1)
      return(out)
    }

    H1 <- outer(X = xi, Y = xi, h1)

    int <- t(wi) %*% H1 %*% wi
    return(int)
  }

  eq2 <- function(gamma, k, alpha, mu, b, sigma, xi, wi) {
    h2 <- function(x, y) {
      out <- h(gamma, k, alpha, mu, b, sigma, x, y, 2)
      return(out)
    }

    H2 <- outer(X = xi, Y = xi, h2)

    int <- t(wi) %*% H2 %*% wi
    return(-1 + k + int)
  }

  eq3 <- function(gamma, k, alpha, mu, b, sigma, xi, wi) {
    h3 <- function(x, y) {
      out <- h(gamma, k, alpha, mu, b, sigma, x, y, 3)
      return(out)
    }

    H3 <- outer(X = xi, Y = xi, h3)

    int <- t(wi) %*% H3 %*% wi
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
