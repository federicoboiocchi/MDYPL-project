# Tests 
rm(list=ls())
# No intercept case 

# Time elapsed comparison

#source("C://Users//andre//Downloads//se1.R")
#source("C://Users//andre//Downloads//solve_se.R")
#source("C://Users//andre//Downloads//se0.R")
#source("C://Users//andre//Downloads//helper-functions.R")

#library("trust")
#library("statmod")
#library("nleqslv")
#library("tictoc")
#library("viridis")

# defining the grids

size <- 10
k <- seq(0.01,0.95,length.out=size)

g <- seq(0,20,length.out=size)
a <- 1/(1+k)
trust_iter <- 0
start <- c(0.5,1,1)

kappa <- rep(k,times=size)
gamma <- rep(g,each=size)
alpha <- rep(a,times=size)

size_sq <- size^2
times <- numeric(size_sq)


for(j in 1:size_sq){
    tic()
    out<-solve_se(kappa[j], gamma[j], alpha[j], intercept = NULL, start, gh = NULL, prox_tol = 1e-10, transform = FALSE, trust_iter = trust_iter)
    elaps <- toc()
    times[j]<- as.numeric(elaps$toc-elaps$tic)
}

times_matrix <- matrix(data=times,ncol=size,nrow=size) 

filled.contour(k, g, times_matrix,
               color.palette = viridis,
               xlab = "kappa", ylab = "gamma",
               main = "Elapsed time")

times_julia <- read.csv("times_j.csv")$times

times_julia_mat <- matrix(data=times_julia,ncol=size,nrow=size) 

filled.contour(k, g, times_julia_mat,
               color.palette = viridis,
               xlab = "kappa", ylab = "gamma",
               main = "Elapsed time (julia)")

