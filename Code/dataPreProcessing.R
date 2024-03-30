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
  
  #split Year range into Period Start and Period End
  impact_data <- impact_data %>%
    separate("Year.Range..", into = c("Period Start", "Period End"), sep = "-") 
  
  #convert Maximum Torino Scale to numeric
  unique(impact_data$Torino.Scale..max..)
  
  #convert chr to numeric value
  impact_data$Torino.Scale..max.. <- as.numeric(impact_data$Torino.Scale..max..)
  impact_data$`Period Start` <- as.numeric(impact_data$`Period Start`)
  impact_data$`Period End` <- as.numeric(impact_data$`Period End`)
  
  #add period field based on period start and end
  impact_data$Period <- impact_data$`Period End` - impact_data$`Period Start`
  
  #since there are two values of Torino scale 0 and (*), 
  #I will convert (*) to NA since (*) is also a notation used to denote Torino Scale
  impact_data$Torino.Scale..max..[impact_data$Torino.Scale..max.. == "(*)" | is.na(impact_data$Torino.Scale..max..)] <- NA
  
  #convert chr to numeric value for Asteroid Magnitude
  orbit_data$Asteroid.Magnitude <- as.numeric(orbit_data$Asteroid.Magnitude)
  
  #missing values
  paste("Percentage of null valies in impact data:", (sum(is.na(impact_data)))*100/nrow(impact_data),"%")
  paste("Percentage of null valies in orbit data:", (sum(is.na(orbit_data)))*100/nrow(orbit_data),"%")
  
  write.csv(impact_data, "./Data/processed_impacts.csv")
  write.csv(orbit_data, "./Data/processed_orbits.csv")
}

processAsteroid("./Data/impacts_data_sentry.csv","./Data/orbits.csv")
