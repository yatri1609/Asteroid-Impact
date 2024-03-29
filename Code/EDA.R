library(dplyr)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(plotly)
library(corrplot)
library(webshot)

#import processed files
data_impacts <- read.csv("./Data/processed_impacts.csv", header=TRUE)
data_orbits <- read.csv("./Data/processed_orbits.csv", header=TRUE)

str(data_impacts)
head(data_impacts)

str(data_orbits)
head(data_orbits)

##### Asteroid features by object name #####
data_impacts_long <- data_impacts %>%
  mutate("Asteroid.Diameter..km.*100" = Asteroid.Diameter..km.*100) %>%
  select(Object.Name,"Asteroid.Diameter..km.*100",Asteroid.Magnitude,Asteroid.Velocity) %>%
  pivot_longer(cols = -Object.Name, names_to = "Metric", values_to = "Value")

AsteroidFeaturesByObject <- plot_ly(data_impacts_long, x = ~Value, y = ~reorder(Object.Name, Value), 
        color = ~Metric, type = 'bar', orientation = 'h') %>%
  layout(yaxis = list(title = 'Asteroids'),
         xaxis = list(title = 'Metrics'),
         barmode = 'stack',
         title = 'Asteroid features by Object',
         legend = list(title = list(text = 'Metric')))

AsteroidFeaturesByObject

##### Asteroid Impact Risk by Year #####
asteroid_counts <- data_impacts %>%
  group_by(Period.End) %>%
  summarise(count = n())

# Plot the stacked bar chart
plot_ly(asteroid_counts, x = ~Period.End, y = ~count, type = 'bar',
        colors = blues9) %>%
  layout(title = 'Asteroid Impact Risk by Year',
         xaxis = list(title = 'Year'),
         yaxis = list(title = 'Count of Asteroids'),
         colorway = RColorBrewer::brewer.pal(8, "Set2"))


##### Average Cumulative Impact Probability by Period Start #####
average_impact <- data_impacts %>%
  group_by(Period.Start) %>%
  summarise(Average_Cumulative_Impact_Probability = mean(Cumulative.Impact.Probability, na.rm = TRUE))

plot_ly(average_impact, x = ~Period.Start, y = ~Average_Cumulative_Impact_Probability, 
        type = 'scatter', mode = 'lines+markers') %>%
  layout(title = "Average Cumulative Impact Probability by Year",
         xaxis = list(title = "Year"),
         yaxis = list(title = "Average Cumulative Impact Probability"))

##### Possible Impacts vs Asteroid features #####
plot_ly(data_impacts, x=~Asteroid.Velocity, y=~Possible.Impacts,
        type='scatter', mode='markers') %>%
  layout(title = "Possible Impacts by Asteroid Velocity",
         xaxis = list(title = "Asteroid Velocity in km/s"),
         yaxis = list(title = "Possible Impacts"))

plot_ly(data_impacts, x=~Asteroid.Magnitude, y=~Possible.Impacts,
        type='scatter', mode='markers') %>%
  layout(title = "Possible Impacts by Asteroid Magnitude",
         xaxis = list(title = "Asteroid Magnitude"),
         yaxis = list(title = "Possible Impacts"))

plot_ly(data_impacts, x=~Asteroid.Diameter..km., y=~Possible.Impacts,
        type='scatter', mode='markers') %>%
  layout(title = "Possible Impacts by Asteroid Diameter",
         xaxis = list(title = "Asteroid Diameter in km"),
         yaxis = list(title = "Possible Impacts"))

##### Orbit data classification #####
asteroid_category_count <- data_orbits %>%
  group_by(Object.Classification) %>%
  summarise(count = n())

plot_ly(asteroid_category_count, x=~Object.Classification, y=~count, type='bar') %>%
  layout(title = "Asteroids per Classification",
         xaxis = list(title = "Asteroid Classification"),
         yaxis = list(title = "Number of Asteroids"))
