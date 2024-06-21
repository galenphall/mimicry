# Libraries
library(ggplot2)

# Parameters
k <- 15  # Length of bitstring
n_predators <- 100  # Initial number of predators
n_mimics <- 100  # Initial number of mimics
n_models <- 100  # Initial number of venomous models

max_mutations_rate <- 0.10  # Maximum mutation rate for mimics
prey_encounter <- 0.02  # Rate at which prey is encountered

natality_predator <- 0.2
mortality_predator <- 0.2
death_threshold <- 0.7

natality_mimic <- 0.2
mortality_mimic <- 0.2
mating_mimic <- 0.2
brood_mimic <- 2

natality_model <- 0.2
mortality_model <- 0.2
mating_model <- 0.2
brood_model <- 2

# Function to generate signals (bit strings)
generate_signals <- function(n, k, mutation_rate = NULL) {
  signals <- matrix(nrow = n, ncol = k)
  for (i in 1:k) {
    if (is.null(mutation_rate)) {
      signals[, i] <- sample(0:1, n, replace = TRUE, prob = runif(2))
    } else {
      signals[, i] <- sample(0:1, n, replace = TRUE, prob = c(1 - mutation_rate, mutation_rate))
    }
  }
  signals <- as.data.frame(signals)
  signals$Phenotype <- rowSums(signals)
  return(signals)
}

# Function to calculate distance between predator and prey
calculate_distance <- function(predator, prey, k) {
  distance <- abs(as.numeric(predator[1:k]) - as.numeric(prey[1:k]))
  return(distance)
}

# Softmax function for calculating probabilities
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  return(exp_x / sum(exp_x))
}

# Function to determine predation probability
predation_probability <- function(predator, prey, risk_tolerance, k) {
  distance <- calculate_distance(predator, prey, k)
  if (all(!is.na(distance))) {
    similarity <- 1 - mean(distance, na.rm = TRUE)
    return(softmax(similarity / risk_tolerance))
  } else {
    return(0)
  }
}

# Function to handle encounter and predation decisions
encounter_and_decide <- function(predators, prey, risk_tolerance, k) {
  predation_outcomes <- data.frame(Predator = integer(), Prey = integer(), Eaten = logical(), Died = logical())
  
  for (i in 1:nrow(predators)) {
    if (runif(1) < prey_encounter && nrow(prey) > 0) {
      j <- sample(1:nrow(prey), 1)
      distance <- calculate_distance(predators[i, ], prey[j, ], k)
      
      if (all(!is.na(distance))) {
        predation_prob <- predation_probability(predators[i, ], prey[j, ], risk_tolerance[i], k)
        will_eat <- runif(1) < predation_prob
        
        is_venomous <- prey[j, "ID"] == "Model"
        will_die <- is_venomous && (runif(1) < (prey[j, "Phenotype"] / k))
        
        predation_outcomes <- rbind(predation_outcomes, 
                                    data.frame(Predator = i, Prey = j, Eaten = will_eat, Died = will_die))
      }
    }
  }
  return(predation_outcomes)
}

# Function to handle reproduction based on natality rate and mutation
reproduce <- function(population, natality_rate, brood_size, mutation_rate = NULL) {
  n_new <- round(nrow(population) * natality_rate)
  new_population <- data.frame()
  
  if (n_new > 0) {
    for (i in 1:n_new) {
      parent1 <- population[sample(1:nrow(population), 1), ]
      parent2 <- population[sample(1:nrow(population), 1), ]
      
      offspring <- data.frame()
      for (j in 1:(ncol(parent1) - 2)) { # Exclude last two columns: Phenotype and ID
        if (!is.null(mutation_rate) && runif(1) < mutation_rate) {
          offspring[j] <- sample(0:1, 1)
        } else {
          offspring[j] <- sample(c(parent1[j], parent2[j]), 1)
        }
      }
      offspring$Phenotype <- sum(as.numeric(offspring[1:(ncol(offspring) - 1)]))
      offspring$ID <- parent1$ID
      
      new_population <- rbind(new_population, offspring)
    }
  }
  
  return(new_population)
}

# Initialize populations
set.seed(123)
signal_model <- generate_signals(n_models, k)
signal_model$ID <- "Model"

signal_mimics <- generate_signals(n_mimics, k, max_mutations_rate)
signal_mimics$ID <- "Mimic"

signal_predators <- generate_signals(n_predators, k)
signal_predators$ID <- "Predator"
risk_tolerance <- rnorm(n_predators, 0.4, 0.2)

# Main simulation loop
time_steps <- 50
for (t in 1:time_steps) {
  cat("Time step:", t, "\n")
  
  # Combine models and mimics for predation
  combined_prey <- rbind(signal_model, signal_mimics)
  
  # Encounters and predation
  predation_outcomes <- encounter_and_decide(signal_predators, combined_prey, risk_tolerance, k)
  
  # Remove eaten prey
  eaten_prey <- unique(predation_outcomes$Prey[predation_outcomes$Eaten])
  if (length(eaten_prey) > 0 && nrow(combined_prey) > 0) {
    combined_prey <- combined_prey[-eaten_prey, ]
  }
  
  # Split combined prey back into models and mimics
  signal_model <- combined_prey[combined_prey$ID == "Model", ]
  signal_mimics <- combined_prey[combined_prey$ID == "Mimic", ]
  
  # Remove dead predators
  dead_predators <- unique(predation_outcomes$Predator[predation_outcomes$Died])
  if (length(dead_predators) > 0 && nrow(signal_predators) > 0) {
    signal_predators <- signal_predators[-dead_predators, ]
  }
  
  # Mortality
  if (nrow(signal_model) > 1) {
    signal_model <- signal_model[sample(nrow(signal_model), size = max(1, round(nrow(signal_model) * (1 - mortality_model)))), ]
  }
  if (nrow(signal_mimics) > 1) {
    signal_mimics <- signal_mimics[sample(nrow(signal_mimics), size = max(1, round(nrow(signal_mimics) * (1 - mortality_mimic)))), ]
  }
  if (nrow(signal_predators) > 1) {
    signal_predators <- signal_predators[sample(nrow(signal_predators), size = max(1, round(nrow(signal_predators) * (1 - mortality_predator)))), ]
  }
  
  # Reproduction
  new_models <- reproduce(signal_model, natality_model, brood_model)
  new_mimics <- reproduce(signal_mimics, natality_mimic, brood_mimic, max_mutations_rate)
  new_predators <- reproduce(signal_predators, natality_predator, 1)
  
  # Add new offspring to populations
  if (nrow(new_models) > 0) {
    signal_model <- rbind(signal_model, new_models)
  }
  if (nrow(new_mimics) > 0) {
    signal_mimics <- rbind(signal_mimics, new_mimics)
  }
  if (nrow(new_predators) > 0) {
    signal_predators <- rbind(signal_predators, new_predators)
  }
  
  # Print population sizes at each time step
  cat("Models:", nrow(signal_model), "Mimics:", nrow(signal_mimics), "Predators:", nrow(signal_predators), "\n")
}

# Visualize the final phenotype distribution
combined_data <- rbind(signal_model, signal_mimics, signal_predators)
colnames(combined_data) <- c(as.character(1:k), "Phenotype", "ID")

ggplot(combined_data, aes(x = Phenotype, fill = ID)) + 
  geom_histogram(binwidth = 1, position = "dodge") +
  labs(title = "Population Phenotype Distribution", x = "Phenotype", y = "Frequency") +
  theme_minimal()
