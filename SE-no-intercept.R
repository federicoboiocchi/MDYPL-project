## Ioannis Kosmidis, 20250729 (based on refactoring non-vectorized code in eqns_GH.R by Federico Boiocchi)

#' The functions in the MDYPL state evolution equations.
#'
#' @param mu aggregate bias parameter.
#' @param b parameter `b` in the state evolution functions.
#' @param sigma square root of the aggregate variance of the MDYPL
#'     estimator.
#' @param kappa asymptotic ratio of columns/rows of the design
#'     matrix. `kappa` should be in `(0, 1)`.
#' @param gamma the square root of the limit of the variance of the
#'     linear predictor.
#' @param alpha the shrinkage parameter of the MDYPL
#'     estimator. `alpha` should be in `(0, 1)`.
#' @param gh A list with the Gauss-Hermite quadrature nodes and
#'     nweights, as returned from `statmod::gauss.quad()` with `kind =
#'     "hermite"`. Default is `NULL`, in which case `gh` is set to
#'     `statmod::gauss.quad(200, kind = "hermite")` is used.
#' @param prox_tol tolerance for the computation of the proximal
#'     operator; default is `1e-10`.
#'
#' @export

mdypl_se <- function(mu, b, sigma, kappa, gamma, alpha, gh = NULL, prox_tol = 1e-10) {
    if (is.null(gh))
        gh <- gauss.quad(200, kind = "hermite")
    a_frac <- 0.5 * (1 + alpha)
    xi <- gh$nodes
    wi <- gh$weights
    n_quad <- length(xi)
    q1 <- rep(sqrt(2) * gamma * xi, times = n_quad)
    q2 <- mu * q1 + rep(sqrt(2 * kappa) * sigma * xi, each = n_quad)
    w2 <- rep(wi, times = n_quad) * rep(wi, each = n_quad)

    plogis2 <- function(x) 1 / (1 + exp(-x))

    ## Solving the fixed-point iteration using Newton Raphson (vectorized)
    prox <- function(x, b) {
        u <- 0
        g0 <- x - b / 2
        while (!all(abs(g0) < prox_tol)) {
            pr <- plogis2(u)
            g0 <- x - u - b * pr
            u <- u + g0 / (b * pr * (1 - pr) + 1)
        }
        u
    }

    p_q1 <- plogis2(q1)
    p_prox <- plogis2(prox(q2 + a_frac * b, b))
    w2p <- 2 * w2 * p_q1  / pi
    prox_resid <- a_frac - p_prox
    out <- c(sum(w2p * q1 * prox_resid),
             1 - kappa - sum(w2p / (1 + b * p_prox * (1 - p_prox))),
             kappa^2 * sigma^2 - b^2 * sum(w2p * prox_resid^2))
    out
}

## More transperent in terms of what is going on.
##
## mdypl_se <- function(mu, b, sigma, iota, kappa, gamma, alpha, theta0, gh = NULL, prox_tol = 1e-10) {
##     if (is.null(gh))
##         gh <- gauss.quad(50, kind = "hermite")
##     xi <- gh$nodes
##     wi <- gh$weights
##     n_quad <- length(xi)
##     xi1 <- rep(xi, times = n_quad)
##     xi2 <- rep(xi, each = n_quad)
##     w2 <- rep(wi, times = n_quad) * rep(wi, each = n_quad)

##     ## Solving the fixed-point iteration using Newton Raphson (vectorized)
##     prox <- function(x, b) {
##         u <- 0
##         g0 <- x - b / 2
##         while (!all(abs(g0) < prox_tol)) {
##             pr <- 1 / (1 + exp(-u))
##             g0 <- x - u - b * pr
##             u <- u + g0 / (b * pr * (1 - pr) + 1)
##         }
##         u
##     }

##     plogis2 <- function(x) (1 + exp(-x))^(-1)
##     a_frac <- 0.5 * (1 + alpha)

##     q1 <- sqrt(2) * gamma * xi1
##     q2 <- sqrt(2) * (mu * gamma * xi1 + sqrt(kappa) * sigma * xi2)
##     p_q1 <- plogis2(q1)
##     p_prox <- plogis2(prox(q2 + a_frac * b, b))
##     w2p <- w2 * p_q1 / pi
##     out <- c(sum(w2p * q1 * (a_frac - p_prox)),
##              1 - kappa - 2 * sum(w2p / (1 + b * p_prox * (1 - p_prox))),
##              kappa^2 * sigma^2 - 2 * b^2 * sum(w2p * (a_frac - p_prox)^2))
##     out
## }



#' Solving the MDYPL state evolution equations.
#'
#' @inheritParams mdypl_se
#' @param start starting values for `mu`, `b`, and `sigma`.
#' @param transform if `TRUE` (default), the optimization is with
#'     respect to `log(mu)`, `log(b)` and `log(sigma)`. If `FALSE`,
#'     then it is over `mu`, `b`, `sigma`. The solution is returned in
#'     terms of the latter three, regardless of how optimization took
#'     place.
#' @param ... further arguments to be passed to `nleqslv::nleqslv()`.
#' @export
#'
solve_mdypl_se <- function(kappa, gamma, alpha, start, gh = NULL, prox_tol = 1e-10, transform = TRUE, ...) {
    if (transform) {
        g <- function(pars) {
            pars <- exp(pars)
            mdypl_se(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
        }
        start <- log(start)
    } else {
        g <- function(pars) {
            mdypl_se2(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
        }
    }
    res <- nleqslv(start, g, ...)
    soln <- if (transform) exp(res$x) else res$x
    attr(soln, "funcs") <- res$fvec
    attr(soln, "iter") <- res$iter
    soln
}
