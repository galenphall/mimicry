library(pracma)  # For norm() function in similarity

generate <- function(num_predators, num_venomous_prey, num_mimics, dim=2, venom_const=0.5, risk_tol_scale=0.1) {
  # Generate random initial positions for predators, venomous prey, and mimics
  detectors <- matrix(runif(num_predators * dim, -2, 2), nrow=num_predators, ncol=dim)
  venomous_signals <- matrix(runif(num_venomous_prey * dim, -2, 2), nrow=num_venomous_prey, ncol=dim)
  mimic_signals <- matrix(runif(num_mimics * dim, -2, 2), nrow=num_mimics, ncol=dim)
  
  # Combine venomous and mimic signals
  signals <- rbind(venomous_signals, mimic_signals)
  
  # Generate risk tolerances for predators
  risk_tols <- rexp(num_predators, rate=1/risk_tol_scale)
  
  # Set venom levels: constant for venomous prey, zero for mimics
  venom_levels <- c(rep(venom_const, num_venomous_prey), rep(0, num_mimics))

  return(list(detectors=detectors, signals=signals, risk_tols=risk_tols, venom_levels=venom_levels))
}

similarity <- function(detectors, signals, phenotype_type='vector', periodic_boundary=NULL) {
  if (phenotype_type == 'vector') {
    if (is.null(periodic_boundary)) {
      dist <- apply(signals, 1, function(s) apply(detectors, 1, function(d) norm(d - s, type="2")))
      dist <- t(dist)  # Transpose to match Python's output shape
    } else {
      dist <- pairwise_periodic_distances_optimized(detectors, signals, periodic_boundary)
    }
    return(-dist^2)
  } else if (phenotype_type == 'bitstring') {
    d <- ncol(signals)
    hamming_distances <- apply(signals, 1, function(s) apply(detectors, 1, function(d) sum(d != s)))
    hamming_distances <- t(hamming_distances)  # Transpose to match Python's output shape
    return(1 - hamming_distances/d)
  } else {
    stop("Not implemented")
  }
}

pairwise_periodic_distances_optimized <- function(v1, v2, b) {
  v1 <- array(v1, dim=c(dim(v1)[1], 1, dim(v1)[2]))
  v2 <- array(v2, dim=c(1, dim(v2)[1], dim(v2)[2]))
  
  diff <- abs(v1 - v2)
  adjusted_diff <- pmin(diff, 2*b - diff)
  
  distances <- sqrt(apply(adjusted_diff^2, c(1,2), sum))
  
  return(distances)
}

calculate_preference_matrix <- function(detectors, signals, risk_tols, phenotype_type='vector', periodic_boundary=NULL) {
  similarity_matrix <- similarity(detectors, signals, phenotype_type, periodic_boundary)
  return(1 - exp(similarity_matrix / outer(risk_tols^2, rep(1, ncol(similarity_matrix)))))
}

calculate_predation_matrix <- function(detectors, signals, risk_tols, handling_time, 
                                       attack_freq, R, phenotype_type='vector', 
                                       periodic_boundary=NULL) {
  preference_matrix <- calculate_preference_matrix(detectors, signals, risk_tols, phenotype_type, periodic_boundary)
  n_predators <- nrow(preference_matrix)
  n_prey <- ncol(preference_matrix)
  n_effective_prey <- rowSums(preference_matrix) + R
  intake_rates <- attack_freq / (1 + n_predators + attack_freq * handling_time * n_effective_prey)
  return(list(
    predation_matrix = intake_rates * preference_matrix,
    n_effective_prey = n_effective_prey
  ))
}

sample_predators <- function(predation_matrix, venom_levels, pred_conversion_ratio, attack_rate, handling_time, R, n_effective_prey, death_rate=0.5) {
  num_predators <- nrow(predation_matrix)
  fitnesses <- rowSums(predation_matrix * (1 - venom_levels) * pred_conversion_ratio - predation_matrix * venom_levels)
  fitnesses <- fitnesses + 1 - death_rate 
  fitnesses <- fitnesses + attack_rate * R / (1 + num_predators + attack_rate * handling_time * n_effective_prey) * pred_conversion_ratio
  means <- pmax(fitnesses, 0)
  counts <- rpois(num_predators, means)
  return(rep(1:num_predators, counts))
}

sample_prey <- function(predation_matrix, popcap, venom_levels, r=0.6) {
  nv <- sum(venom_levels > 0)
  nm <- sum(venom_levels == 0)
  num_prey <- ifelse(venom_levels > 0, nv, nm)
  fitnesses <- r * (1 - num_prey / popcap) - colSums(predation_matrix)
  means <- pmax(fitnesses, 0)
  counts <- rpois(nv + nm, means)
  return(rep(1:(nv + nm), counts))
}

phenotype_mutate <- function(phenotypes, mutation_rate=0.01, phenotype_type='vector') {
  if (phenotype_type == 'vector') {
    return(phenotypes + matrix(rnorm(length(phenotypes), sd=mutation_rate), nrow=nrow(phenotypes)))
  } else if (phenotype_type == 'bitstring') {
    stop("Not implemented")
  } else {
    stop("Not implemented")
  }
}

impose_periodic_boundary <- function(vectors, boundary=5) {
  if (is.null(boundary)) {
    return(vectors)
  }
  
  vectors <- (vectors + boundary) %% (2 * boundary) - boundary
  
  return(vectors)
}

                                                             
update <- function(detectors, signals, risk_tols, venom_levels, num_venomous, R, r_R, k_R,
                   r_prey, handling_time, attack_rate, predator_conversion_ratio, prey_popcap,
                   mutation_rate, phenotype_type='vector', periodic_boundary=NULL, mutate_venom=FALSE, mutate_risk=FALSE) {
  
  # Calculate predation matrix
  pred_result <- calculate_predation_matrix(
    detectors, signals, risk_tols,                  
    handling_time,                  
    attack_rate,       
    R,
    phenotype_type=phenotype_type, 
    periodic_boundary=periodic_boundary
  )
  predation_matrix <- pred_result$predation_matrix
  n_effective_prey <- pred_result$n_effective_prey

  # Sample next generation of predators and prey
  predator_children <- sample_predators(
    predation_matrix, 
    venom_levels, 
    predator_conversion_ratio, 
    attack_rate, 
    handling_time, 
    R,
    n_effective_prey
  )
  
  prey_children <- sample_prey(
    predation_matrix, 
    prey_popcap, 
    venom_levels,
    r_prey
  )

  # Update resource availability
  num_predators <- nrow(predation_matrix)
  delta_R <- sum(attack_rate * R / (1 + num_predators + attack_rate * handling_time * n_effective_prey))
  R <- R + r_R * R * (1 - R / k_R) - delta_R
  R <- max(R, 0)

  # Get phenotypes of children
  predator_childrens_detectors <- detectors[predator_children, , drop=FALSE]
  prey_childrens_signals <- signals[prey_children, , drop=FALSE]
  
  # Mutate phenotypes
  predator_childrens_detectors <- phenotype_mutate(
    predator_childrens_detectors, 
    mutation_rate=mutation_rate, 
    phenotype_type=phenotype_type)
  
  prey_childrens_signals <- phenotype_mutate(
    prey_childrens_signals, 
    mutation_rate=mutation_rate, 
    phenotype_type=phenotype_type)

  # Apply periodic boundary conditions
  predator_childrens_detectors <- impose_periodic_boundary(predator_childrens_detectors, periodic_boundary)
  prey_childrens_signals <- impose_periodic_boundary(prey_childrens_signals, periodic_boundary)

  # Optionally mutate risk tolerances and venom levels
  if (mutate_risk) {
    predator_childrens_risk_tols <- phenotype_mutate(
      risk_tols[predator_children], 
      mutation_rate=mutation_rate, 
      phenotype_type=phenotype_type)
    predator_childrens_risk_tols <- abs(predator_childrens_risk_tols)
  } else {
    predator_childrens_risk_tols <- risk_tols[predator_children]
  }

  if (mutate_venom) {
    prey_childrens_venoms <- phenotype_mutate(
      venom_levels[prey_children], 
      mutation_rate=mutation_rate, 
      phenotype_type=phenotype_type)
    
    prey_childrens_venoms[prey_childrens_venoms > 0.9999] <- 0.9999
    prey_childrens_venoms[prey_childrens_venoms < 0.0001] <- 0.0001
    prey_childrens_venoms[venom_levels[prey_children] == 0] <- 0
  } else {
    prey_childrens_venoms <- venom_levels[prey_children]
  }

  new_num_venomous <- sum(prey_childrens_venoms > 0)

  return(list(
    detectors = predator_childrens_detectors, 
    signals = prey_childrens_signals, 
    risk_tols = predator_childrens_risk_tols, 
    venom_levels = prey_childrens_venoms, 
    num_venomous = new_num_venomous, 
    R = R
  ))
}