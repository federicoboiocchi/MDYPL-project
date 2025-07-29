## Ioannis Kosmidis, 20250729 (based on previous code by Federico Boiocchi)

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
#'     `statmod::gauss.quad(50, kind = "hermite")` is used.
#' @param prox_tol tolerance for the computation of the proximal
#'     operator; default is `1e-10`.
#'
#' @export
mdypl_se <- function(mu, b, sigma, kappa, gamma, alpha, gh = NULL, prox_tol = 1e-10) {
    if (is.null(gh))
        gh <- gauss.quad(50, kind = "hermite")

    ## Solving the fixed-point iteration using Newton Raphson (vectorized)
    prox <- function(x, b) {
        u <- 0
        g0 <- x - b / 2
        while (!all(abs(g0) < prox_tol)) {
            pr <- 1 / (1 + exp(-u))
            g0 <- x - u - b * pr
            u <- u + g0 / (b * pr * (1 - pr) + 1)
        }
        u
    }

    xi <- gh$nodes
    wi <- gh$weights
    n_quad <- length(xi)
    xi1 <- rep(xi, times = n_quad)
    xi2 <- rep(xi, each = n_quad)
    w2 <- rep(wi, times = n_quad) * rep(wi, each = n_quad)

    sd_zs <- sqrt(mu^2 * gamma^2 + kappa * sigma^2)
    rho <- mu * gamma / sd_zs
    B <- 2 * (1 - rho^2)
    sqrtB <- sqrt(B)
    log_jac <- log(B * gamma * sd_zs)
    log_C <- - log(2 * pi * gamma * sd_zs * sqrt(1 - rho^2))
    a_frac <- (1 + alpha) / 2

    log_pdf <- log_C + log_jac + 2 * rho * xi1 * xi2
    z <- xi1 * sqrtB * gamma
    zs <- xi2 * sqrtB * sd_zs
    prox_val <- prox(zs + a_frac * b, b)
    p_z <- plogis(z)
    p_prox_val <- plogis(prox_val)
    v <- p_z * exp(log_pdf) * w2
    c(2 * sum(v * xi1 * (a_frac - p_prox_val)),
      2 * sum(v / (1 + b * p_prox_val * (1 - p_prox_val))) - 1 + kappa,
      2 * sum(v * (a_frac - p_prox_val)^2) * b^2 / kappa^2 - sigma^2)
}

#' Solving the MDYPL state evolution equations.
#' @inheritParams mdypl_se
#' @param ... further arguments to be passed to `nleqslv::nleqslv()`.
#' @export
solve_mdypl_se <- function(kappa, gamma, alpha, start, gh = NULL, prox_tol = 1e-10, ...) {
    g <- function(pars) {
        mdypl_se(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
    }
    res <- nleqslv(start, g, ...)
    soln <- res$x
    attr(soln, "funcs") <- res$fvec
    attr(soln, "iter") <- res$iter
    soln
}
