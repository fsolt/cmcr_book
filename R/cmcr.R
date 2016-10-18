library(tidyverse)
library(rstan)
library(beepr)

load("data/ls2015_dat.Rdata")

ls <- dat %>%
    mutate_at(vars(keith, hc, ciri), funs(. + 1)) %>%  # lowest score now 1
    select(country, year, 
           keith, hc, ciri, xconst, cim.cat,
           fvfacto.cat, laworder.cat, gcr.cat) %>% 
    gather(variable, score, -country, -year) %>% 
    filter(!is.na(score)) %>% 
    group_by(country) %>%
    mutate(cc_rank = n(),         # number of country-year-items (data-richness)
           firstyr = first(year, order_by = year),
           lastyr = last(year, order_by = year)) %>%
    ungroup() %>%
    arrange(desc(cc_rank), country, year) %>% # order by data-richness
    # Generate numeric codes for countries, years, questions, and question-cuts
    mutate(ccode = as.numeric(factor(country, levels = unique(country))),
           tcode = as.integer(year - min(year) + 1),
           qcode = as.numeric(factor(variable, levels = unique(variable)))) %>%
    arrange(ccode, tcode, qcode)
    
    
### Delete these when turning into a function
seed <- 324
iter <- 40
chains <- 4
cores <- chains
x <- ls
###

S <- x %>% 
    group_by(qcode) %>%
    summarize(S = max(score) - 1) %>% 
    select(S) %>%
    unlist() %>% 
    as.numeric()

stan_data <- list(  K = max(x$ccode),
                    T = max(x$tcode),
                    Q = max(x$qcode),
                    N = length(x$score),
                    kk = x$ccode,
                    tt = x$tcode,
                    qq = x$qcode,
                    y = x$score,
                    S = S,
                    MS = max(S))


start <- proc.time()
out1 <- stan(file = "R/cmcr.stan",
             data = stan_data,
             seed = seed,
             iter = iter,
             cores = cores,
             chains = chains,
             control = list(max_treedepth = 20,
                            adapt_delta = .8))
runtime <- proc.time() - start
runtime

lapply(get_sampler_params(out1, inc_warmup = FALSE),
       summary, digits = 2)

#Chime
beep()

# Plot of convergence statistics
ex_summary <- as.data.frame(summary(out1)[[1]])
ex_summary$parameter <- as.factor(gsub("\\[.*]", "", rownames(ex_summary)))
ggplot(ex_summary) + 
    aes(x = parameter, y = Rhat, color = parameter) + 
    geom_jitter(height = 0, width = 0.5, show.legend = FALSE) + 
    ylab(expression(hat(italic(R))))
