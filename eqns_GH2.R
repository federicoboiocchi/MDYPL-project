#' eqns_GH function that gives the value of the three
#' equations of the system once given mu, b, sigma.
#'
#' @param gh A list with the Gauss-Hermite quadrature points, as
#'     output from `statmod::gauss.quad()` with `kind =
#'     "hermite"`. Default is `NULL`, in which case
#'     `statmod::gauss.quad(50, kind = "hermite")` is used.
#' @param k is the ratio #covariates / #observations as #obs. and p
#'     goes to +Inf
#' @param gamma standard deviation of the linear predictor as #obs
#'     goes to +Inf

eqns_GH2 <- function(mu, b, sigma, kappa, gamma, alpha, gh = NULL, prox_tol = 1e-10) {
    if (is.null(gh))
        gh <- gauss.quad(50, kind = "hermite")
    
    xi <- gh$nodes
    wi <- gh$weights
    n_quad <- length(xi)
    xi1 <- rep(xi, times = n_quad)
    xi2 <- rep(xi, each = n_quad)
    w2 <- rep(wi, times = n_quad) * rep(wi, each = n_quad)
    
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


solve_mDYPL_SE <- function(kappa, gamma, alpha, start, gh = NULL, prox_tol = 1e-10, ...) {
    g <- function(pars) {
        eqns_GH2(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
    }
    res <- nleqslv(start, g, ...)
    soln <- res$x
    attr(soln, "funcs") <- res$fvec
    attr(soln, "iter") <- res$iter
    soln
}
