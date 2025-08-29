# Figure 1

rm(list=ls())
library(paletteer)

library("statmod")
library("nleqslv")
library("R.utils")

project_path <- "C:/Users/andre/Documents/brglm21/R"
#project_path <- "~/brglm2/R"
source(file.path(project_path, "solve_se.R"))
source(file.path(project_path, "se.R"))
source(file.path(project_path, "utils.R"))
source(file.path(project_path, "mdyplFit.R"))

par(mfrow=c(1,1))
# inverse logit
sigma <- function(x) 1 / (1 + exp(-x))

# transition curve k = h(gamma)

k <- function(gamma, n = 5000) {

  x <- rnorm(n)
  p <- sigma(gamma * x)

  y <- ifelse(runif(n) < p, 1, -1)

  v <- y * x
  z <- rnorm(n)

  # The order of the operation is: first max(x,0) then the square and
  # then the expected value (approximated via montecarlo)
  # I use pmax because I want the comparison between components of vectors

  obj <- function(t) {
    mean(pmax(t * v - z, 0)^2 )
  }

  # minmization over t in c(-5,5), using optimize

  opt <- optimize(obj, interval = c(-5, 5))
  return(opt$objective)
}

#gamma_vals <- seq(0, 20, length.out = 100)

# compute the means of multiple wiggly transition curves
# in order to have a smooth final estimate. it is worth observing
# that the output values of h are random and not deterministic

# k_vals <- apply(replicate(150,sapply(gamma_vals,k)),1,mean)

# Plot

# plot(k_vals,gamma_vals,type = "l",main="Phase transition curve",
#     xlab=expression(k),ylab=expression(gamma))

smooth <-100

# Boundary function
f <- function(gamma){
  out <- apply(replicate(smooth,sapply(gamma,k)),1,mean)
}

# x values for plotting
gamma <- seq(0, 30, length.out = 150)
xmin <- 0
xmax <- 1

output <- f(gamma)

plot(NA,xlim = c(xmin, xmax),ylim = c(min(gamma), max(gamma)),
     xlab = expression(k), ylab = expression(gamma),main="ML estimate transition curve")

polygon(c(rep(xmin, length(gamma)), rev(output)),c(gamma, rev(gamma))
        ,col = adjustcolor("orange", alpha.f = 0.3), border = NA)

polygon(c(output, rep(xmax, length(gamma))),c(gamma, rev(gamma)),
        col = adjustcolor("blue", alpha.f = 0.3), border = NA)

lines(output,gamma, col = "grey50", lwd = 1.5)

# Simulations of logitic regression estimation 
# via MDYPL fit across different settings of kappa and gamma


set.seed(123)
n <- 1000
gamma <- c(1,5,10,15,20)
k <- c(0.1,0.25,0.5,0.75,0.8)
kg <- expand.grid(gamma,k)
size <- dim(kg)[1]
col1 <- adjustcolor("blue", alpha.f = 0.8)
col2 <- adjustcolor("darkorange",alpha.f=0.5)


for(i in 1:size){
  k <- kg[i,2]
  p <- n*k
  ga <- kg[i,1]
  alpha <- 1/(1+k)
  timeout <- 3
  se_pars <- NULL
  while(is.null(se_pars)){
    
    X <- matrix(rnorm(n * p, 0, 1), nrow = n, ncol = p) 
    X_std <- scale(X,center=TRUE,scale=TRUE)
    betas0 <- rep(c(-3, -3/2, 0, 3/2, 3), each = p / 5)
    betas <- ga * betas0 / sqrt(sum(betas0^2))
    probs <- plogis(drop(X %*% betas))
    y <- rbinom(n, 1, probs)
    fit_mdypl <- glm(y ~ -1 + X, family = binomial(), method = "mdyplFit") 
    b_est <- coef(fit_mdypl)
    
    se_pars <- tryCatch(withTimeout(solve_se(kappa = k, ss = ga, alpha = 1/(1+k),
                                             start = c(0.5,1,1),
                                             corrupted = FALSE, gh = NULL, prox_tol = 1e-10,
                                             transform = FALSE, init_method = "Nelder-Mead",
                                             init_iter = 50),timeout = timeout,onTimeout = "error"),TimeoutException=function(ex){message("Timeout exceeded (", timeout, "s). Restarting...")
                                               return(NULL)
                                             }
    )
  }
  mu <- se_pars[1]
  beta_res <- b_est/mu
  
  par(mfrow = c(1, 2))
  pf <-p/5
  
  means <- c(mean(beta_res[1:(pf)]),mean(beta_res[(pf+1):(2*pf)]),mean(beta_res[(2*pf+1):(3*pf)]),
             mean(beta_res[(3*pf+1):(4*pf)]),mean(beta_res[(4*pf+1):p]))
  means <- rep(means,each=p/5)
  
  means_not_res <- c(mean(b_est[1:(pf)]),mean(b_est[(pf+1):(2*pf)]),mean(b_est[(2*pf+1):(3*pf)]),
                     mean(b_est[(3*pf+1):(4*pf)]),mean(b_est[(4*pf+1):p]))
  means_not_res <- rep(means_not_res,each=p/5)
  
  minbeta <- min(betas)
  maxbeta <- max(betas)
  plot(betas, type = "s",
       main = NULL,
       xlab = NA, ylab = NA,lwd=2.5,xlim=c(-5,p+5),ylim=c(minbeta+1/2*minbeta,maxbeta+1/2*maxbeta),
       xaxt="n",yaxt="n")
  points(b_est, col = NA, bg = col2, pch = 21)
  points(c(1:p),means_not_res, col = col1,type="s",lwd=2.5)
  legend("topleft", legend = as.expression(bquote(k == .(k))), bty = "n")
  legend("topleft", legend = as.expression(bquote(gamma == .(ga))),
         bty="n", inset=c(0, 0.08))
  
  plot(betas, type = "s",
       main = NULL,
       xlab = NA, ylab = NA,lwd=2.5,xaxt="n",yaxt="n",xlim=c(-5,p+5),ylim=c(minbeta+1/2*minbeta,maxbeta+1/2*maxbeta))
  points(beta_res, col = NA, bg = col2, pch = 21)
  lines(c(1:p),means, col = col1, type="s",pch = 21,lwd=2.5)
  legend("topleft", legend = as.expression(bquote(k == .(k))), bty = "n")
  legend("topleft", legend = as.expression(bquote(gamma == .(ga))),
         bty="n", inset=c(0, 0.08))
}

