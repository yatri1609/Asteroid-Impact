#import libraries
library(tidyr)
library(dplyr)
library(ggplot2)
library(corrplot)

processAsteroid <- function(impact_data_file, orbit_data_file){
  #read data from csv
  impact_data <- read.csv(impact_data_file, header=TRUE)
  orbit_data <- read.csv(orbit_data_file, header=TRUE)
  
  #column names of the dataset
  colnames(impact_data)
  colnames(orbit_data)
  
  #Data transformations
  
  #data structure
  str(impact_data)
  str(orbit_data)
  
  #convert Maximum Torino Scale to numeric
  unique(impact_data$Maximum.Torino.Scale)
  
  #convert chr to numeric value
  impact_data$Maximum.Torino.Scale <- as.numeric(impact_data$Maximum.Torino.Scale)
  
  #add period field based on period start and end
  impact_data$Period <- impact_data$Period.End - impact_data$Period.Start
  
  #since there are two values of Torino scale 0 and (*), 
  #I will convert (*) to NA since (*) is also a notation used to denote Torino Scale
  impact_data$Maximum.Torino.Scale[impact_data$Maximum.Torino.Scale == "(*)" | is.na(impact_data$Maximum.Torino.Scale)] <- NA
  
  #convert chr to numeric value for Asteroid Magnitude
  orbit_data$Asteroid.Magnitude <- as.numeric(orbit_data$Asteroid.Magnitude)
  
  #missing values
  paste("Number of null valies in impact data:", (sum(is.na(impact_data)))*100/nrow(impact_data),"%")
  paste("Number of null valies in orbit data:", (sum(is.na(orbit_data)))*100/nrow(orbit_data),"%")
  
  write.csv(impact_data, "./Data/processed_impacts.csv")
  write.csv(orbit_data, "./Data/processed_orbits.csv")
}

processAsteroid("./Data/impacts.csv","./Data/orbits.csv")
