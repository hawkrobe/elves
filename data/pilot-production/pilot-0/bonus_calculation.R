library(here)

raw_data_path <- here("data/pilot-production/pilot-0/raw_data")
processed_data_path <- here("data/pilot-production/pilot-0/processed_data")

extract_participant_points <- function(raw_data_file){
  raw_data <- read_csv(raw_data_file)
  participant_data <- raw_data |> 
    select(prolific_id, total_points) |> 
    filter(!is.na(total_points)) |> 
    group_by(prolific_id) |> 
    summarize(max_points = round(max(total_points), 2))
  
  return(participant_data)
}

pilot_data <- grep("inprog", list.files(raw_data_path, full.names = TRUE), value = TRUE, invert = TRUE)

d_participants <- do.call(rbind, lapply(pilot_data, extract_participant_points))

d_participants |> write_csv(here(processed_data_path, "prolific_payments.csv"))
