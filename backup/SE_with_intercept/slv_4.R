# solver for the system of four non linear equations (intercept included)

# arguments:
# n_app: number of nodes (total combination of nodes will be n^2)
# lim_opt: abs value of the limit used to solve the optim. problem
# in the proximal operator function
# start: starting value for the numerical solver
# maxit: maximum number of iterations for the solver
# trace: verbose output optional
# app_met: approximation method


slv_4 <- function(k, gamma, alpha, t0, n_app, lim_opt, start, maxit, trace, app_met,coord_trasf) {
  ## # library
  ##   library(nleqslv)

  # select the approx method
  eqn_func <- switch(app_met,
                     "cub" = function(mu, b, sigma, iota) eqns4(gamma, k, alpha, mu, b, sigma, iota, t0, lim_opt, coord_trasf, method="cub", n_app),
                     "GH"  = function(mu, b, sigma, iota) eqns4(gamma, k, alpha, mu, b, sigma, iota, t0, lim_opt, coord_trasf, method="GH", n_app),
                     stop("Invalid approximation method")
  )

  # objective function
  f <- function(par) {
    mu <- par[1]
    b <- par[2]
    sigma <- par[3]
    iota <- par[4]
    out <- eqn_func(mu, b, sigma,iota)
    return(out)
  }

  control_list <- list(trace = trace, maxit = maxit)
  solution<- nleqslv(start, f, control = control_list)$x
  return(solution)
}
