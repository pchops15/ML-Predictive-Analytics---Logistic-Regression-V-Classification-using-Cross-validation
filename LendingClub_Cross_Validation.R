# ------------------------------------------------------------------------------ 
# 4-FOLD CROSS VALIDATION + HOLDOUT EVALUATION
# ------------------------------------------------------------------------------

library(tidyverse)
library(rpart)
library(rpart.plot)
library(MLmetrics)

# ------------------------------------------------------------------------------
# LOAD IN DATA

setwd("~/Desktop/414/LendingClub") # CHANGE THE WORKING DIRECTORY

lc_data <- read.csv("small_lending_club_clean.csv")

# ------------------------------------------------------------------------------
# CLEANING AND MUTATING DATA

# Ensuring that loan_status only includes values that Figel highlighted 
lc_data <- lc_data %>%
  filter(loan_status %in% c(
    "Fully Paid", "Charged Off", "Default", 
    "Late (31-120 days)", "Late (16-30 days)", "Current"
  ))


lc_data <- lc_data %>%
  mutate(
    loan_status = if_else(loan_status %in% c("Charged Off", "Default"), 
                          "Default", "Repaid"),
    loan_status = factor(loan_status, levels = c("Repaid", "Default")),
    fico = (fico_range_low + fico_range_high) / 2,
    income = ifelse(application_type == "Individual",
                    annual_inc, annual_inc_joint),
    combined_dti = ifelse(application_type == "Individual",
                          dti, dti_joint)
  )


# Selecting a subset of useful features
model_data <- lc_data %>%
  select(purpose, int_rate, loan_amnt, verification_status, fico, combined_dti,
         loan_status, sub_grade, income)

set.seed(42)

# ---- Split into work (80%) and holdout (20%) --------------------------------
num_rows <- nrow(model_data)
rows_to_move <- sample(1:num_rows, size = ceiling(0.2 * num_rows))
holdout_set <- model_data[rows_to_move, ]
work_set <- model_data[-rows_to_move, ]

# ---- 4-FOLD CROSS VALIDATION ------------------------------------------------
k <- 4
row_indices <- sample(1:nrow(work_set))
folds <- cut(1:nrow(work_set), breaks = k, labels = FALSE)

cart_logloss <- numeric(k)
logit_logloss <- numeric(k)

for (i in 1:k) {
  cat("Running Fold", i, "\n")
  
  test_indices <- row_indices[folds == i]
  train_data <- work_set[-test_indices, ]
  test_data  <- work_set[test_indices, ]
  
  # --- CART ------------------------------------------------------------------
  trained_cart_model <- rpart(
    loan_status ~ ., 
    data = train_data, 
    method = "class",
    maxdepth = 4
  )
  
  predicted_cart <- predict(trained_cart_model, newdata = test_data, type = "prob")
  predicted_cart_probs <- predicted_cart[, "Default"]
  actual_01 <- ifelse(test_data$loan_status == "Default", 1, 0)
  cart_logloss[i] <- LogLoss(y_pred = predicted_cart_probs, y_true = actual_01)
  
  # --- LOGISTIC REGRESSION ---------------------------------------------------
  trained_log_model <- glm(
    loan_status ~ ., 
    data = train_data, 
    family = "binomial"
  )
  
  predicted_log <- predict(trained_log_model, newdata = test_data, type = "response")
  logit_logloss[i] <- LogLoss(y_pred = predicted_log, y_true = actual_01)
}

mean_cart_logloss <- mean(cart_logloss)
mean_logit_logloss <- mean(logit_logloss)

cat("\nAverage 4-Fold LogLoss (CART):", mean_cart_logloss, "\n")
cat("Average 4-Fold LogLoss (Logistic):", mean_logit_logloss, "\n\n")


# ------------------------------------------------------------------------------ 
# FINAL HOLDOUT EVALUATION: ACCURACY + CONFUSION MATRIX
# ------------------------------------------------------------------------------

# --- CART MODEL --------------------------------------------------------------
final_cart_model <- rpart(
  loan_status ~ ., 
  data = work_set,
  method = "class",
  maxdepth = 4
)

cart_pred_class <- predict(final_cart_model, newdata = holdout_set, type = "class")
cart_pred_prob  <- predict(final_cart_model, newdata = holdout_set, type = "prob")[, "Default"]
cart_accuracy <- Accuracy(cart_pred_class, holdout_set$loan_status)
cart_conf_matrix <- ConfusionMatrix(cart_pred_class, holdout_set$loan_status)

cart_logloss_holdout <- LogLoss(
  y_pred = cart_pred_prob,
  y_true = ifelse(holdout_set$loan_status == "Default", 1, 0)
)

cat("CART Holdout LogLoss:", cart_logloss_holdout, "\n")
cat("CART Holdout Accuracy:", cart_accuracy, "\n")
cat("CART Confusion Matrix:\n")
print(cart_conf_matrix)
cat("\n")


# --- LOGISTIC REGRESSION MODEL ----------------------------------------------
final_log_model <- glm(
  loan_status ~ ., 
  data = work_set,
  family = "binomial"
)

log_pred_prob <- predict(final_log_model, newdata = holdout_set, type = "response")
log_pred_class <- ifelse(log_pred_prob > 0.5, "Default", "Repaid")
log_pred_class <- factor(log_pred_class, levels = c("Repaid", "Default"))

log_accuracy <- Accuracy(log_pred_class, holdout_set$loan_status)
log_conf_matrix <- ConfusionMatrix(log_pred_class, holdout_set$loan_status)

log_logloss_holdout <- LogLoss(
  y_pred = log_pred_prob,
  y_true = ifelse(holdout_set$loan_status == "Default", 1, 0)
)

cat("Logistic Regression Holdout LogLoss:", log_logloss_holdout, "\n")
cat("Logistic Regression Holdout Accuracy:", log_accuracy, "\n")
cat("Logistic Regression Confusion Matrix:\n")
print(log_conf_matrix)
cat("\n")
