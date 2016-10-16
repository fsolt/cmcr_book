library(rstan)

# Attach the example dataset. The ltm package is required.
data(Science, package = "ltm")

# Convert dataset to an integer matrix with values 1 ... 4
M <- matrix(NA, ncol = ncol(Science), nrow = nrow(Science))
for (i in 1:ncol(M)) M[, i] <- as.integer(Science[, i])

# Assemble data list for Stan
ex_list <- list( I = ncol(M), 
                 J = nrow(M), 
                 N = length(M), 
                 ii = rep(1:ncol(M), each = nrow(M)), 
                 jj = rep(1:nrow(M), times = ncol(M)), 
                 y = as.vector(M), 
                 K = 1, 
                 W = matrix(1, nrow = nrow(M), ncol = 1) )

# Run Stan model
ex_fit <- stan(file = "R/grsm_latent_reg.stan", 
               data = ex_list, 
               chains = 4, 
               cores = 4,
               iter = 1200,
               seed = 324)

# Plot of convergence statistics
ex_summary <- as.data.frame(summary(ex_fit)[[1]])
ex_summary$Parameter <- as.factor(gsub("\\[.*]", "", rownames(ex_summary)))
ggplot(ex_summary) + 
    aes(x = Parameter, y = Rhat, color = Parameter) + 
    geom_jitter(height = 0, width = 0.5, show.legend = FALSE) + 
    ylab(expression(hat(italic(R))))

# View table of parameter posteriors
print(ex_fit, pars = c("alpha", "beta", "kappa", "lambda"))