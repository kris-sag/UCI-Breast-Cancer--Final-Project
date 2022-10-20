# exploratory analysis

library(dplyr)
library(magrittr)

cancer = read.csv("wdbc.data",header=FALSE)
names = c("ID_Number", "Diagnosis", 
    "Mean_Radius", "Mean_Texture", "Mean_Permieter", "Mean_Area", "Mean_Smoothness", "Mean_Compactness", "Mean_Concavity", "Mean_Concave_Points", "Mean_symmetry", "Mean_Fractal_Dimension",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE", "Compactness_SE", "Concavity_SE", "Concave_Points_SE", "symmetry_SE", "Fractal_Dimension_SE",
    "Worst_Radius", "Worst_Texture", "Worst_Perimeter", "Worst_Area", "Worst_Smoothness", "Worst_Compactness", "Worst_Concavity", "Worst_Concave_Points", "Worst_symmetry", "Worst_Fractal_Dimension")
colnames(cancer) <- names

cancer %>% group_by(Diagnosis)  %>% summarize(n(), 
    mean(Mean_Radius), 
    mean(Mean_Texture),
    mean(Mean_Permieter),
    mean(Mean_Area),
    mean(Mean_Smoothness),
    mean(Mean_Compactness),
    mean(Mean_Concavity),
    mean(Mean_Concave_Points),
    mean(Mean_symmetry),
    mean(Mean_Fractal_Dimension))

cancer %>% group_by(Diagnosis) %>% summarize(n(),
    mean(Radius_SE),
    mean(Texture_SE),
    mean(Perimeter_SE),
    mean(Area_SE),
    mean(Smoothness_SE),
    mean(Compactness_SE),
    mean(Concavity_SE),
    mean(Concave_Points_SE),
    mean(symmetry_SE),
    mean(Fractal_Dimension_SE))

# cancer %>% group_by(Diagnosis) %>% summarize(n(),
#     mean(Worst_Radius), 
#     mean(Worst)



malignant = cancer %>% filter(Diagnosis == "M")
benign = cancer %>% filter(Diagnosis == "B")

par(mfrow=c(1,2))
boxplot(malignant$Mean_Radius)
boxplot(benign$Mean_Radius)

par(mfrow=c(1,2))
boxplot(malignant$Mean_Fractal_Dimension)
boxplot(benign$Mean_Fractal_Dimension)

# training and classification