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
## mdypl_se <- function(mu, b, sigma, kappa, gamma, alpha, gh = NULL, prox_tol = 1e-10) {
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

##     mu_gamma2 <- (mu * gamma)^2
##     sigma2 <- sigma^2
##     var_zs <- mu_gamma2 + kappa * sigma2
##     rho2 <- mu_gamma2 / var_zs
##     sqrt_Bh <- sqrt(1 - rho2)
##     a_frac <- 0.5 * (1 + alpha)

##     log_pdf <- log(sqrt_Bh / pi) + 2 * sqrt(rho2) * xi1 * xi2
##     z <- xi1 * sqrt(2) * sqrt_Bh * gamma
##     zs <- xi2 * sqrt(2) * sqrt_Bh * sqrt(var_zs)
##     p_z <- plogis2(z)
##     p_prox_val <- plogis2(prox(zs + a_frac * b, b))
##     v <- w2 * p_z * exp(log_pdf)
##     c(2 * sum(v * xi1 * (a_frac - p_prox_val)),
##       2 * sum(v / (1 + b * p_prox_val * (1 - p_prox_val))) - 1 + kappa,
##       2 * sum(v * (a_frac - p_prox_val)^2) * (b / kappa)^2 - sigma2)
## }

mdypl_se <- function(mu, b, sigma, kappa, gamma, alpha, gh = NULL, prox_tol = 1e-10) {
    if (is.null(gh))
        gh <- gauss.quad(50, kind = "hermite")
    xi <- gh$nodes
    wi <- gh$weights
    n_quad <- length(xi)
    xi1 <- rep(xi, times = n_quad)
    xi2 <- rep(xi, each = n_quad)
    w2 <- rep(wi, times = n_quad) * rep(wi, each = n_quad) / pi

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

    plogis2 <- function(x) (1 + exp(-x))^(-1)
    a_frac <- 0.5 * (1 + alpha)

    q1 <- sqrt(2) * gamma * xi1
    q2 <- sqrt(2) * (mu * gamma * xi1 + sqrt(kappa) * sigma * xi2)
    p_q1 <- plogis2(q1)
    p_prox <- plogis2(prox(q2 + a_frac * b, b))
    w2p <- w2 * p_q1
    out <- c(sum(w2p * q1 * (a_frac - p_prox)),
             1 - kappa - 2 * sum(w2p / (1 + b * p_prox * (1 - p_prox))),
             kappa^2 * sigma^2 - 2 * b^2 * sum(w2p * (a_frac - p_prox)^2))
    cat("max|q1| =", max(abs(q1)), "max|q2| =", max(abs(q2)), "\n")
    cat("max(|out|)", max(abs(out)), "\n")
    out
}



#' Solving the MDYPL state evolution equations.
#' @inheritParams mdypl_se
#' @param start starting values for `mu`, `b`, and `sigma`.
#' @param ... further arguments to be passed to `nleqslv::nleqslv()`.
#' @export
#'
solve_mdypl_se <- function(kappa, gamma, alpha, start, gh = NULL, prox_tol = 1e-10, transform = TRUE, ...) {
    if (transform) {
        g <- function(pars) {
            pars <- exp(pars)
            mdypl_se(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
        }
    } else {
        g <- function(pars) {
            mdypl_se2(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
        }
        start <- log(start)
    }
    res <- nleqslv(start, g, ...)
    soln <- if (transform) exp(res$x) else res$x
    attr(soln, "funcs") <- res$fvec
    attr(soln, "iter") <- res$iter
    soln
}
