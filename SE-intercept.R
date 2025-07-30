#' Federico Boiocchi 20250729 (based on the vectorization design in the script eqns_GH2.R for MDYPL SE equations eq by Ioannis Kosmidis)
#'
#' @param mu aggregate bias parameter.
#' @param b parameter `b` in the state evolution functions.
#' @param sigma square root of the aggregate variance of the MDYPL
#'     estimator.
#' @param iota limits of the MDYPL estimate for the intercept as the sample size goes to +Inf
#' @param kappa asymptotic ratio of columns/rows of the design
#'     matrix. `kappa` should be in `(0, 1)`.
#' @param gamma the square root of the limit of the variance of the
#'     linear predictor.
#' @param alpha the shrinkage parameter of the MDYPL
#'     estimator. `alpha` should be in `(0, 1)`.
#' @param intercept intercept of the logistic regression model
#' @param gh A list with the Gauss-Hermite quadrature nodes and
#'     nweights, as returned from `statmod::gauss.quad()` with `kind =
#'     "hermite"`. Default is `NULL`, in which case `gh` is set to
#'     `statmod::gauss.quad(n_appx, kind = "hermite")` is used.
#' @param prox_tol tolerance for the computation of the proximal
#'     operator; default is `1e-10`. fixed point problem solved via Newton-Raphson
#' @param n_appx number of Gauss-Hermite nodes used

mdypl_se4 <- function(mu, b, sigma, iota, kappa, gamma, alpha, intercept, gh = NULL, prox_tol = 1e-10, n_appx = 50) {
  if (is.null(gh)) {
    gh <- gauss.quad(n_appx, kind = "hermite")
  }

  xi <- gh$nodes
  wi <- gh$weights

  n_quad <- length(xi)
  xi1 <- rep(xi, times = n_quad)
  xi2 <- rep(xi, each = n_quad)
  w2 <- rep(wi, times = n_quad) * rep(wi, each = n_quad)

  # proximal operator function via Newton-Raphson

  plogis <- function(x) (1 + exp(-x))^(-1)

  prox <- function(x, b) {
    u <- 0
    g0 <- x - b / 2
    while (!all(abs(g0) < prox_tol)) {
      pr <- plogis(u)
      g0 <- x - u - b * pr
      u <- u + g0 / (b * pr * (1 - pr) + 1)
    }
    u
  }

  sqrt2 <- sqrt(2)

  q1 <- xi1 * sqrt2

  z <- gamma * q1 + intercept
  zs <- mu * gamma * xi1 * sqrt2 + sqrt(kappa) * sigma * xi2 * sqrt2 + iota
  p_z <- plogis(z)
  p_z_n <- plogis(-z)
  a_frac <- 0.5 * (1 + alpha)

  prox_val <- prox(a_frac * b + zs, b)
  p_prox_val <- plogis(prox_val)

  prox_val_n <- prox(a_frac * b - zs, b)
  p_prox_val_n <- plogis(prox_val_n)

  prox_resid <- a_frac - p_prox_val
  prox_resid_n <- a_frac - p_prox_val_n

  # four equations of the nonlinear system

  v <- w2 * (1 / pi)
  out <- c(
    sum(v * z * (p_z * (prox_resid) - p_z_n * (prox_resid_n))),
    sum(v * ((p_z / (1 + b * p_prox_val * (1 - p_prox_val))) + (p_z_n / (1 + b * p_prox_val_n * (1 - p_prox_val_n))))) - 1 + kappa,
    sum(v * ((p_z * (prox_resid)^2) + p_z_n * (prox_resid_n)^2)) * (b / kappa)^2 - sigma^2,
    sum(v * ((p_z * (prox_resid)) - (p_z_n * (prox_resid_n))))
  )
  out
}

#' Solving the MDYPL state evolution equations.
#'
#' @inheritParams mdypl_se4
#' @param start starting values for `mu`, `b`,`sigma`,and `iota`.
#' @param transform if `TRUE` (default), the optimization is with
#'     respect to `log(mu)`, `log(b)`,`log(sigma)`,and `log(iota)`. If `FALSE`,
#'     then it is over `mu`, `b`, `sigma` and `iota`. The solution is returned in
#'     terms of the latter four, regardless of how optimization took
#'     place.
#' @param ... further arguments to be passed to `nleqslv::nleqslv()`.
#' @export
#'

solve_mdypl_se4 <- function(kappa, gamma, alpha, intercept, start, gh = NULL, prox_tol = 1e-10, n_appx = 50, transform = TRUE, ...) {
  no_int <- 1:3
  if (transform) {
    g <- function(pars) {
      pars[no_int] <- exp(pars[no_int])
      mdypl_se4(mu = pars[1], b = pars[2], sigma = pars[3], iota = pars[4], kappa = kappa, gamma = gamma, alpha = alpha, intercept = intercept, gh = gh, prox_tol = prox_tol, n_appx = n_appx)
    }
    start[no_int] <- log(start[no_int])
  } else {
    g <- function(pars) {
      mdypl_se4(mu = pars[1], b = pars[2], sigma = pars[3], iota = pars[4], kappa = kappa, gamma = gamma, alpha = alpha, intercept = intercept, gh = gh, prox_tol = prox_tol, n_appx = n_appx)
    }
  }
  res <- nleqslv(start, g, ...)
  soln <- if (transform) c(exp(res$x[no_int]), res$x[4]) else res$x
  attr(soln, "funcs") <- res$fvec
  attr(soln, "iter") <- res$iter
  soln
}
