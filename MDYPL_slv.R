# solver for the system of three non linear equations

# arguments:
# n: number of nodes (total combination of nodes will be n^2)
# lim_opt: abs value of the limit used to solve the optim. problem
# in the proximal operator function
# start: starting value for the numerical solver
# maxit: maximum number of iterations for the solver
# trace: verbose output optional
# app_met: approximation method
# opt_met: optimization method

MDYPL_slv <- function(k, gamma, alpha, n, lim_opt, start, maxit, trace, app_met, opt_met) {
  # libraries

  library(rootSolve)
  library(BB)
  library(nleqslv)

  # for the method cubature (cuhre) the absolute value of the upper and lower limit is
  # set to 200 (arbirarily chosen)

  # select the approx method
  eqn_func <- switch(app_met,
    "cubature" = function(mu, b, sigma) eqns_cub(mu, b, sigma, k, gamma, alpha, lim_opt, lim = 200),
    "GH"       = function(mu, b, sigma) eqns_GH(mu, b, sigma, k, gamma, alpha, n, lim_opt),
    "GH-vec"   = function(mu, b, sigma) eqns_GH_vec(mu, b, sigma, k, gamma, alpha, n, lim_opt),
    stop("Invalid approximation method")
  )

  # objective function
  f <- function(par) {
    mu <- par[1]
    b <- par[2]
    sigma <- par[3]
    out <- eqn_func(mu, b, sigma)
    return(out)
  }


  control_list <- list(trace = trace, maxit = maxit)

  # select the optimizer
  solution <- switch(opt_met,
    "nleqslv" = {
      res <- nleqslv(start, f, control = control_list)
      res$x
    },
    "BB" = {
      res <- BBsolve(start, f, control = control_list)
      res$par
    },
    "rootSolve" = {
      res <- multiroot(f, start, maxiter = maxit)
      res$root
    },
    stop("Invalid optimization method")
  )
  return(solution)
}
