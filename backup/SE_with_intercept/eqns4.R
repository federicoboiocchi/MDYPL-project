# 4 equations non linear system with intercept

## rm(list = ls())

## n <- 60 # number of nodes for GH cubature
## mu <- 2
## b <- 1
## sigma <- 3
## iota <- 1.7
## t0 <- 1.1

## lim_opt <- 1000 # boundary for the optimization problem that computes the proximal operator

## gamma <- 4
## k <- 0.3
## alpha <- 1 / (1 + k)

# coord_trasf: TRUE or FALSE , it performs an additional change of variables in order to integrate over [0,1]x[0,1] instead
# of R^2 (if combined with method="GH" it doesn't have any effect). It should be used with method="cub"

# method = "GH" : Gaussian-Hermite cubature through nested 1D quadrature (matrix-vector product is used instead of double for cycles)
# method = "cub": method cuhre from cubintegrate function cubature package



eqns4 <- function(gamma, k, alpha, mu, b, sigma, iota, t0, lim_opt, coord_trasf, method, n) {

  # inverse link function
  link <- function(x) {
    1 / (1 + exp(-x))
  }

  # first derivative of inv link function
  link2 <- function(x) {
    exp(x) / ((1 + exp(x))^2)
  }

  # proximal operator
  prox <- function(x, b) {
    f <- function(u) {
      b * log(1 + exp(u)) + (1 / 2) * (x - u)^2
    }
    opt <- optimize(f, maximum = FALSE, interval = c(-lim_opt, lim_opt))
    out <- opt$minimum
    return(out)
  }
  prox <- Vectorize(prox, vectorize.args = "x")


  Q <- function(alpha, b, z) {
    a_frac <- (1 + alpha) / 2
    ab_frac <- b * (1 + alpha) / 2
    out <- a_frac - link(prox(z + (ab_frac), b))
    return(out)
  }


  g_func1 <- function(z1, z2) {
    link(z1) * z1 * Q(alpha, b, z2) - link(-z1) * z1 * Q(alpha, b, -z2)
  }

  g_func2 <- function(z1, z2) {
    ab_frac <- b * (1 + alpha) / 2
    out <- link(z1) / (1 + b * link2(prox(z2 + ab_frac, b))) + link(-z1) / (1 + b * link2(prox(ab_frac - z2, b))) - 1 + k
    return(out)
  }

  g_func3 <- function(z1, z2) {
    (link(z1) * Q(alpha, b, z2)^2 + link(-z1) * Q(alpha, b, -z2)^2) * (b^2) / (k^2) - sigma^2
  }

  g_func4 <- function(z1, z2) {
    (link(z1) * Q(alpha, b, z2) - link(-z1) * Q(alpha, b, -z2))
  }

  # h: function to integrate for "cub" or to integrate numerically via "GH"

  h <- function(x, y, pos) {

    jac <- 1

    z <- x
    g <- y

    if (coord_trasf == TRUE & method == "cub") {
      u <- x * (1 - x)
      v <- y * (1 - y)

      x <- (2 * x - 1) / u
      y <- (2 * y - 1) / v

      jac <- (1 - 2 * u) * (1 - 2 * v) / ((u * v)^2)

      z <- x
      g <- y
    }

    pdf <- dnorm(z) * dnorm(g)

    if (method == "GH") {
      z <- x * sqrt(2)
      g <- y * sqrt(2)
      pdf <- 1 / (pi)
    }

    z1 <- gamma * z + t0
    z2 <- mu * gamma * z + sqrt(k) * sigma * g + iota

    out <- pdf * jac * switch(pos,
      g_func1(z1, z2),
      g_func2(z1, z2),
      g_func3(z1, z2),
      g_func4(z1, z2)
    )
    return(out)
  }

  if (coord_trasf == TRUE & method == "cub") {
    lowlim <- 0
    uplim <- 1
  } else {
    lowlim <- -Inf
    uplim <- +Inf
  }

  if (method == "cub") {
    library(cubature)

    h <- function(x, pos) {
      out <- h(x[1], x[2], pos)
      return(out)
    }

    eq1 <- cubintegrate(function(x) h(x, 1), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral
    eq2 <- cubintegrate(function(x) h(x, 2), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral
    eq3 <- cubintegrate(function(x) h(x, 3), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral
    eq4 <- cubintegrate(function(x) h(x, 4), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral

    return(c(eq1, eq2, eq3, eq4))
  }

  if (method == "GH") {
    library(statmod)

    res <- gauss.quad(n, kind = "hermite")
    xx <- res$nodes
    w <- matrix(data = res$weights, nrow = n, ncol = 1)

    H1 <- outer(X = xx, Y = xx, function(x, y) h(x, y, 1))
    H2 <- outer(X = xx, Y = xx, function(x, y) h(x, y, 2))
    H3 <- outer(X = xx, Y = xx, function(x, y) h(x, y, 3))
    H4 <- outer(X = xx, Y = xx, function(x, y) h(x, y, 4))

    int1 <- t(w) %*% H1 %*% w
    int2 <- t(w) %*% H2 %*% w
    int3 <- t(w) %*% H3 %*% w
    int4 <- t(w) %*% H4 %*% w

    return(c(int1, int2, int3, int4))
  }
}

## eqns4(gamma, k, alpha, mu, b, sigma, iota, t0, lim_opt, coord_trasf = FALSE, method = "GH", n)

