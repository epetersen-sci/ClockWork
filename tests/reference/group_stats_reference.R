# Reference values for tests/test_group_stats.py, from base R only.
A <- c(12.1, 14.3, 11.8, 15.2, 13.9, 12.7, 14.8, 13.1)
B <- c(16.4, 18.9, 15.2, 17.7, 19.3, 16.8, 18.1)
C <- c(13.5, 21.2, 9.8, 17.4, 24.6, 11.3, 19.9, 15.7, 22.8)
D <- c(14.0, 15.1, 13.2, 16.3, 14.7, 15.8)
g <- list(A=A, B=B, C=C, D=D)
v <- unlist(g); f <- factor(rep(names(g), sapply(g, length)))
fmt <- function(x) formatC(x, digits=12, format="g")
w <- oneway.test(v ~ f, var.equal=FALSE)
cat("welch_F", fmt(w$statistic), "df2", fmt(w$parameter[2]), "p", fmt(w$p.value), "\n")
k <- kruskal.test(v ~ f)
cat("kw_H", fmt(k$statistic), "p", fmt(k$p.value), "\n")
# Games-Howell from its definition (Games & Howell 1976).
nm <- names(g)
for (i in 1:3) for (j in (i+1):4) {
  a <- g[[i]]; b <- g[[j]]; na <- length(a); nb <- length(b)
  se2 <- var(a)/na + var(b)/nb
  q <- abs(mean(a)-mean(b)) / sqrt(se2/2)
  df <- se2^2 / ((var(a)/na)^2/(na-1) + (var(b)/nb)^2/(nb-1))
  cat("gh", nm[i], nm[j], fmt(ptukey(q, 4, df, lower.tail=FALSE)), "\n")
}
# Dunn's test from its definition (Dunn 1964), with tie correction, Holm-adjusted.
r <- rank(v); N <- length(v)
ties <- table(v); tc <- sum(ties^3 - ties) / (12*(N-1))
mr <- tapply(r, f, mean); n <- table(f)
ps <- c(); lab <- c()
for (i in 1:3) for (j in (i+1):4) {
  z <- (mr[i]-mr[j]) / sqrt((N*(N+1)/12 - tc) * (1/n[i] + 1/n[j]))
  ps <- c(ps, 2*pnorm(-abs(z))); lab <- c(lab, paste(nm[i], nm[j]))
}
adj <- p.adjust(ps, "holm")
for (i in seq_along(ps)) cat("dunn", lab[i], fmt(adj[i]), "\n")
t2 <- t.test(A, B, var.equal=FALSE)
cat("welch_t", fmt(t2$statistic), "df", fmt(t2$parameter), "p", fmt(t2$p.value), "\n")
m2 <- wilcox.test(A, B)
cat("mwu_W", fmt(m2$statistic), "p", fmt(m2$p.value), "\n")
