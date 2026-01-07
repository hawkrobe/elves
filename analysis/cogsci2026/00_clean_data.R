# Libraries

library(here)
library(tidyverse)
library(tidyjson)

# Set paths
raw_data_path <- here("data/full-production/raw_data/run-1")
processed_data_path <- here("data/full-production/processed_data/")

# Read and Join Raw Datasets

extract_participant_data <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file, show_col_types = FALSE)
  participant_data <- raw_data |> 
    select(subject_id, prolific_id, frequency_ratio, time_pressure, 
           selected_training_block_id, selected_test_block_id, 
           survey_difficulty:ai_usage_description) |> 
    filter(!is.na(survey_difficulty))
  
  return(participant_data)
}

extract_exposure_data <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file, show_col_types = FALSE)
  exposure_data <- raw_data |> 
    select(subject_id, is_correct, response_text, target, angle, targetFreq, response) |> 
    filter(!is.na(is_correct)) |> 
    mutate(parsed = map(response, jsonlite::parse_json)) |>
    unnest_wider(parsed, names_sep = "_") |>
    filter(!is.na(`parsed_exposure-response`)) |> 
    select(subject_id, is_correct, response_text, target, angle, targetFreq)
  exposure_data
}

extract_training_data <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file, show_col_types = FALSE)
  training_data <- raw_data |> 
    select(subject_id, is_correct:familiarization_trial_number, trial_index, rt) |> 
    filter(!is.na(familiarization_trial_number))
  training_data
}

extract_training_recall_data <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file, show_col_types = FALSE)
  recall_data <- raw_data |> 
    select(subject_id, is_correct:targetFreq, rt, outer_loop_iteration) |> 
    filter(!is.na(outer_loop_iteration), !is.na(is_correct))
  recall_data
}

extract_production_data <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file, show_col_types = FALSE)
  production_data <- raw_data |> 
    select(subject_id, rt, is_correct:targetFreq, trial_id, nearest_trained_angle:timed_out) |> 
    filter(!is.na(nearest_trained_angle))
  production_data
}

extract_final_recall_data <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file, show_col_types = FALSE)
  final_recall_data <- raw_data |> 
    select(subject_id, rt, is_correct:targetFreq, outer_loop_iteration, response) |> 
    filter(!is.na(is_correct), !is.na(response)) |>
    mutate(parsed = map(response, jsonlite::parse_json)) |>
    unnest_wider(parsed, names_sep = "_") |>
    filter(!is.na(`parsed_final-recall-response`)) |> 
    select(subject_id, rt, is_correct:targetFreq, outer_loop_iteration)
  final_recall_data
}

raw_data <- grep("inprog", list.files(raw_data_path, full.names = TRUE), value = TRUE, invert = TRUE)

d_participants <- do.call(rbind, lapply(raw_data, extract_participant_data))
d_training <- do.call(rbind, lapply(raw_data, extract_training_data))
d_recall <- do.call(rbind, lapply(raw_data, extract_training_recall_data))
d_production <- do.call(rbind, lapply(raw_data, extract_production_data))
d_final_recall <- do.call(rbind, lapply(raw_data, extract_final_recall_data))
d_exposure <-  do.call(rbind, lapply(raw_data, extract_exposure_data))

# strip out and store prolific data

d_prolific_mapping <- d_participants |> select(subject_id, prolific_id)
d_participants <- d_participants |> select(-prolific_id)

## Store angle mappings by-participant so we can recover frequency conditions
d_angle_mappings <- d_exposure |> 
  filter(is_correct == 1) |> 
  select(-response_text, -is_correct)

# Fix production data next-nearest angle assignments/distances
# based on actual target angle...

d_production_cleaned <- d_production |> 
  mutate(distance = nearest_trained_angle - angle,
         #fx next-neares angle assignment
         c_next_nearest_angle = ifelse(distance > 0, 
                                               nearest_trained_angle - 45, 
                                               nearest_trained_angle + 45),
         next_nearest_angle = case_when(c_next_nearest_angle < 0 ~ 360+c_next_nearest_angle,
                                        c_next_nearest_angle > 360 ~  c_next_nearest_angle - 360,
                                        TRUE ~ c_next_nearest_angle)) |> 
  select(subject_id, rt, is_correct, trial_id,
         response_text,
         targetAngle = angle,
         targetLabel = target,
         targetFreq,
         nearestAngle = nearest_trained_angle,
         nextNearestAngle = next_nearest_angle,
         distance,
         bonus_points,
         total_points,
         timed_out
  ) |> 
  # join in correct angle mappings based on the angle number
  left_join(d_angle_mappings |> 
              rename(nextNearestAngle = angle,
                     nextNearestFreq = targetFreq,
                     nextNearestLabel = target)) |> 
  left_join(d_angle_mappings |> select(-target) |> 
              rename(nearestAngle = angle,
                     nearestFreq = targetFreq)) |> 
  left_join(d_angle_mappings |> 
              select(subject_id, 
                     response_text = target,
                     responseFreq = targetFreq)) |> 
  # correct condition assignments
  mutate(between_freqs = case_when(nearestFreq == "HF" & nextNearestFreq == "LF" ~ TRUE,
                                   nearestFreq == "LF" & nextNearestFreq == "HF" ~ TRUE,
                                   TRUE ~ FALSE),
         ambig_trial = ifelse(distance <= -11, TRUE, FALSE),
         is_critical = ifelse(between_freqs & ambig_trial, TRUE, FALSE))

# Write to processed_data folder

d_prolific_mapping |> write_csv(here(processed_data_path, "prolific_mapping.csv"))
d_participants |> write_csv(here(processed_data_path, "participants.csv"))
d_training |> write_csv(here(processed_data_path, "training.csv"))
d_recall |> write_csv(here(processed_data_path, "recall.csv"))
d_production_cleaned |> write_csv(here(processed_data_path, "production.csv"))
d_final_recall |> write_csv(here(processed_data_path, "final_recall.csv"))
d_exposure |>  write_csv(here(processed_data_path, "exposure.csv"))

