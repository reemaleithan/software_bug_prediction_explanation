# R snippets for Break-Down with DALEX
# read more about the metod at
# Explanatory Model Analysis
# https://pbiecek.github.io/ema/


# Prepare data
library("DALEX")
jdt_training <- read.csv(file.choose(), header = TRUE)
jdt_testing <- read.csv(file.choose(), header = TRUE)

# Train a model
library(ranger)
set.seed(1313)
#jdt_rf <- ranger(label ~ .,
#                 data = jdt_train,
#                 probability = TRUE,
#                 classification = TRUE)
library(randomForest)
jdt_rf <- randomForest(label ~.,
                       data = jdt_training)
jdt_rf


# Prepare an explainer
library("DALEX")
jdt_ex <- explain(jdt_rf,
                  data  = jdt_training,
                  y     = jdt_training$label == 0,
                  label = "Logistic Regression for Linux_1_instance_id= 343880")

# Prepare an instance
jdt_ins <- data.frame(
  "10019_unlock" = 0,
  "10881_eth" = 0,
  "10114_free" = 0,
  "10028_lock" = 0,
  "10053_pdev" = 0,
  "10638_ifdef" = 0,
  "11078_static" = 10,
  "10441_notic" = 0,
  "403_lines_changed" = 29,
  "10633_desc" = 0,
  "11323_area" = 0,
  "414_lines_inserted" = 48,
  "11623_printk" = 18,
  "11127_share" = 0,
  "1012_firstparm" = 4,
  "10841_window" = 0,
  "1181_CASE" = 0,
  "10042_cmd" = 0,
  "11898_good" = 0,
  "label" =1
)

jdt_ins
jdt_explanation <- predict_parts(explainer = jdt_ex,
                                 new_observation = jdt_ins,
                                 type = "break_down")

predict(jdt_ex, jdt_ins)
plot(jdt_explanation)
