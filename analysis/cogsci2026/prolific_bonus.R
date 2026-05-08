library(here)
library(tidyverse)

raw_data_path <- here("data/full-production/raw_data/exp2-1")
processed_data_path <- here("data/full-production/processed_data")

extract_participant_points <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file)
  participant_data <- raw_data |> 
    select(any_of(c("prolific_id", "total_points"))) |> 
    filter(!is.na(total_points)) |> 
    group_by(prolific_id) |> 
    summarize(max_points = round(max(total_points, na.rm = T), 2))
  return(participant_data)
}

pilot_data <- grep("training_complete", list.files(raw_data_path, full.names = TRUE), value = TRUE, invert = TRUE)

d_participants <- do.call(rbind, lapply(pilot_data, extract_participant_points))

d_participants |> write_csv(here(processed_data_path, "prolific_bonuses/good-enough-production-full-sample-exp2_v3
.csv"))
