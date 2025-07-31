plogis2 <- function(x) 1 / (1 + exp(-x))

## Solving the fixed-point iteration using Newton Raphson (vectorized)
prox <- function(x, b, tol) {
    u <- 0
    g0 <- x - b / 2
    while (!isTRUE(all(abs(g0) < tol))) {
        pr <- plogis2(u)
        g0 <- x - u - b * pr
        u <- u + g0 / (b * pr * (1 - pr) + 1)
    }
    u
}
