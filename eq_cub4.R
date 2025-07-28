# 4 equations non linear system with intercept

rm(list = ls())

# arbitrarily chosen parameters

mu <- 1.5
b <- 5
sigma <- 2.4
iota <- 2.1
t0 <- -0.7  # intercept theta_0

lim_opt <- 1000 # absolute value of the boundary of the univariate optimization problem 
# that has to be solved to compute the proximal operator function.  

gamma <- 10
k <- 0.4
alpha <- 1 / (1 + k)

# coord_trasf could be TRUE or FALSE: it allows the user to choose whether to 
# do an additional transformation and then integrate over [0,1]x[0,1] instead of R^2
# this function transforms the input but also computes the associated jacobian.

# eq_cub4 gives the values of the four equations intercept included. It computes integrals using
# cubintegrate

eq_cub4 <- function(gamma, k, alpha, mu, b, sigma, iota, t0, lim_opt, coord_trasf) {
  link <- function(x) {
    1 / (1 + exp(-x))
  }

  link2 <- function(x) {
    exp(x) / ((1 + exp(x))^2)
  }

  prox <- function(x, b) {
    f <- function(u) {
      b * log(1 + exp(u)) + (1 / 2) * (x - u)^2
    }
    opt <- optimize(f, maximum = FALSE, interval = c(-lim_opt, lim_opt))
    out <- opt$minimum
    return(out)
  }

  # DIFFERENCE in definition of Q: article says 
  
  # Q <- function(alpha, b, z) {
  #   (1/(1 + alpha)) - link(prox(z + (b/(1 + alpha)), b))
  # }
  
  # Julia code uses this different version
  
  Q <- function(alpha, b, z) {
    ((1 + alpha) / 2) - link(prox(z + (b * (1 + alpha) / 2), b))
  }


  g_func1 <- function(z1, z2) {
    link(z1) * z1 * Q(alpha, b, z2) - link(-z1) * z1 * Q(alpha, b, -z2)
  }

  # DIFFERENCE: article says b/(1+alpha)
  # while julia code is b*(1+alpha)/2 
  
  g_func2 <- function(z1, z2) {
    link(z1) / (1 + b * link2(prox(z2 + (b * (1 + alpha) / 2), b))) + link(-z1) / (1 + b * link2(prox(b * (1 + alpha) / 2 - z2, b))) - 1 + k
  }

  g_func3 <- function(z1, z2) {
    (link(z1) * Q(alpha, b, z2)^2 + link(-z1) * Q(alpha, b, -z2)^2) * (b^2) / (k^2) - sigma^2
  }

  g_func4 <- function(z1, z2) {
    (link(z1) * Q(alpha, b, z2) - link(-z1) * Q(alpha, b, -z2))
  }


  h <- function(x, pos) {
    
    jac <- 1
    z <- x[1]
    g <- x[2]
    
    if (coord_trasf == TRUE) {
      u <- x[1] * (1 - x[1])
      v <- x[2] * (1 - x[2])

      x[1] <- (2 * x[1] - 1) / u
      x[2] <- (2 * x[2] - 1) / v

      jac <- (1 - 2 * u) * (1 - 2 * v) / ((u * v)^2)

      z <- x[1]
      g <- x[2]
    }

    z1 <- gamma * z + t0    # DIFFERENCE: article says  z1 <- mu*gamma*z + t0
    z2 <- mu * gamma * z + sqrt(k) * sigma * g + iota

    # DOUBT: why aren't we considering the jacobian of the following transformation
    
    #  z1 <- gamma * z + t0 
    #  z2 <- mu * gamma * z + sqrt(k) * sigma * g + iota
    
    # and instead we consider only the jacobian of the transformation that transforms the domain from 
    # [0,1]x[0,1] in R^2
    
    out <- dnorm(z) * dnorm(g) * jac * switch(pos,
      g_func1(z1, z2),
      g_func2(z1, z2),
      g_func3(z1, z2),
      g_func4(z1, z2)
    )
    return(out)
  }

  if(coord_trasf == TRUE){
    lowlim<-0
    uplim<-1
  } else {
    lowlim<- -Inf
    uplim<- +Inf
  }
  
  eq1 <- cubintegrate(function(x) h(x, 1), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral
  eq2 <- cubintegrate(function(x) h(x, 2), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral
  eq3 <- cubintegrate(function(x) h(x, 3), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral
  eq4 <- cubintegrate(function(x) h(x, 4), lower = c(lowlim, lowlim), upper = c(uplim, uplim), method = "cuhre")$integral

  return(c(eq1, eq2, eq3, eq4))
}

eq_cub4(gamma, k, alpha, mu, b, sigma, iota, t0, lim_opt,coord_trasf=TRUE)
