## Ioannis Kosmidis, 20250731

#' Solve the MDYPL state evolution equations with or without intercept.
#'
#' #' @param mu aggregate bias parameter.
#' @param kappa asymptotic ratio of columns/rows of the design
#'     matrix. `kappa` should be in `(0, 1)`.
#' @param gamma the square root of the limit of the variance of the
#'     linear predictor.
#' @param alpha the shrinkage parameter of the MDYPL
#'     estimator. `alpha` should be in `(0, 1)`.
#' @param intercept if `NULL` (default) then the MDYPL state evolution
#'     equations for the model with no intercept parameter are
#'     solved. If a real then the equations for the models with
#'     intercept parameter equal to `intercept` are solved.
#' @param start a vector with starting values for `mu`, `b`,`sigma`
#'     (and `iota` if `intercept` is numeric).
#' @param gh A list with the Gauss-Hermite quadrature nodes and
#'     nweights, as returned from `statmod::gauss.quad()` with `kind =
#'     "hermite"`. Default is `NULL`, in which case `gh` is set to
#'     `statmod::gauss.quad(200, kind = "hermite")` is used.
#' @param prox_tol tolerance for the computation of the proximal
#'     operator; default is `1e-10`.
#' @param transform if `TRUE` (default), the optimization is with
#'     respect to `log(mu)`, `log(b)`,`log(sigma)`, (and `iota` if
#'     `intercept` is numeric). If `FALSE`, then it is over `mu`, `b`,
#'     `sigma` (and `iota` if `intercept` is numeric). The solution is
#'     returned in terms of the latter set, regardless of how
#'     optimization took place.
#' @param trust_iter how many iterations of `trust::trust()` should we
#'     make to get starting values for `nleqslv::nleqslv()`? Default
#'     is `5`, but can also be `0` in which case `start` is directly
#'     passed to `nleqslv:nleqslv()`.
#' @param ... further arguments to be passed to `nleqslv::nleqslv()`.
#'
#' @details
#'
#' `trust_iter` iterations of `trust::trust()` are used towards
#' minimizing `sum(se)^2`, where se is a vector of the state evolution
#' functions. The solution is then passed to `nleqslv::nleqslv()` for
#' a more aggressive iteration.
#' 
#' @export
solve_se <- function(kappa, gamma, alpha, intercept = NULL, start, gh = NULL, prox_tol = 1e-10, transform = TRUE, trust_iter = 5, ...) {
    if (trust_iter > 0) {
        start <- trust_se(kappa, gamma, alpha, intercept, start, gh, prox_tol, iterlim = trust_iter)
    }
    nleqslv_se(kappa, gamma, alpha, intercept, start, gh, prox_tol, transform, ...)
}


nleqslv_se <- function(kappa, gamma, alpha, intercept = NULL, start, gh = NULL, prox_tol = 1e-10, transform = TRUE, ...) {
    no_intercept <- is.null(intercept)
    if (no_intercept) {
        stopifnot(length(start) == 3)
        if (transform) {
            g <- function(pars) {
                pars <- exp(pars)
                se0(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
            }
            start <- log(start)
        } else {
            g <- function(pars) {
                se0(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol)
            }
        }
    } else {
        stopifnot(length(start) == 4)
        no_int <- 1:3
        if (transform) {
            g <- function(pars) {
                pars[no_int] <- exp(pars[no_int])
                se1(mu = pars[1], b = pars[2], sigma = pars[3], iota = pars[4], kappa = kappa, gamma = gamma, alpha = alpha, intercept = intercept, gh = gh, prox_tol = prox_tol)
            }
            start[no_int] <- log(start[no_int])
        } else {
            g <- function(pars) {
                se1(mu = pars[1], b = pars[2], sigma = pars[3], iota = pars[4], kappa = kappa, gamma = gamma, alpha = alpha, intercept = intercept, gh = gh, prox_tol = prox_tol)
            }
        }
    }
    res <- nleqslv(start, g, ...)
    if (transform) {
        if (no_intercept) {
            soln <- exp(res$x)
        } else {
            soln <- c(exp(res$x[no_int]), res$x[4])
        }
    } else {
        soln <- res$x
    }
    attr(soln, "funcs") <- res$fvec
    attr(soln, "iter") <- res$iter
    soln
}

trust_se <- function(kappa, gamma, alpha, intercept = NULL, start, gh = NULL, prox_tol = 1e-10, ...) {
    ssq <- function(x) sum(x^2)
    no_intercept <- is.null(intercept)
    if (no_intercept) {
        npars <- 3
        g <- function(pars) {
            pars <- exp(pars)
            se0(mu = pars[1], b = pars[2], sigma = pars[3], kappa = kappa, gamma = gamma, alpha = alpha, gh = gh, prox_tol = prox_tol) |> ssq()
        }
        start <- log(start)
    } else {
        npars <- 4
        stopifnot(length(start) == 4)
        no_int <- 1:3
        g <- function(pars) {
            pars[no_int] <- exp(pars[no_int])
            se1(mu = pars[1], b = pars[2], sigma = pars[3], iota = pars[4], kappa = kappa, gamma = gamma, alpha = alpha, intercept = intercept, gh = gh, prox_tol = prox_tol) |> ssq()
        }
        start[no_int] <- log(start[no_int])
    }

    h <- matrix(0, npars, npars)
    upp_inds <- upper.tri(h, diag = TRUE)
    low_inds <- lower.tri(h, diag = TRUE)
    
    vec2mat <- function(vec, d) {
        h <- matrix(NA, npars, npars)
        h[upp_inds] <- vec
        h[low_inds] <- t(h)[low_inds]
        h
    }

    obj <- function(pars) {
        v <- numDeriv::genD(g, pars)        
        list(value = v$f0,
             gradient = v$D[1:npars],
             hessian = vec2mat(v$D[-c(1:npars)]))
    }
    
    res <- trust(obj, start, rinit = 1, rmax = 5, ...) 

    if (no_intercept) {
        soln <- exp(res$argument)
    } else {
        soln <- c(exp(res$argument[no_int]), res$argument[4])
    }

    attr(soln, "objective") <- res$value
    attr(soln, "iter") <- res$iterations
    soln
}
