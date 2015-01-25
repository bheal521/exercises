library(RSQLite)
library(sqldf)
library(ggplot2)
library(pROC)

## Connect to the SQLite database provided ##
RTI_db <- dbConnect(SQLite(), dbname = "C:/Users/Ben/Documents/GitHub/exercises/exercise01/exercise01.sqlite")


## what the heck is in this thing?? ##
dbListTables(RTI_db)

## 9 different tables: countries, education_levels, marital_statuses, occupations, races, records, relationships, sexes, and workclasses ##
## let's take a look at the different tables and their contents ##
## create data frames of all tables ##

countries <- dbReadTable(RTI_db, "countries")
education_levels <- dbReadTable(RTI_db, "education_levels")
marital_statuses <- dbReadTable(RTI_db, "marital_statuses")
occupations <- dbReadTable(RTI_db, "occupations")
races <- dbReadTable(RTI_db, "races")
records <- dbReadTable(RTI_db, "records")
relationships <- dbReadTable(RTI_db, "relationships")
sexes <- dbReadTable(RTI_db, "sexes")
workclasses <- dbReadTable(RTI_db, "workclasses")

## close the connection with the db ##
dbDisconnect(RTI_db)

#######################################################################################################
## 1. Write a SQL query that creates a consolidated dataset from the normalized tables               ##
## in the database. In other words, write a SQL query that "flattens" the database to a single table.##
#######################################################################################################

flat <- sqldf("select a.*, b.name as countries, c.name as education_levels, d.name as marital_status, e.name as occupations, f.name as races, 
                g.name as relationships, h.name as sex, i.name as workclasses
                from records as a 
                left join countries as b ON a.country_id=b.id
                left join education_levels as c ON a.education_level_id = c.id
                left join marital_statuses as d ON a.marital_status_id = d.id
                left join occupations as e ON a.occupation_id = e.id
                left join races as f ON a.race_id = f.id
                left join relationships as g ON a.relationship_id = g.id
                left join sexes as h ON a.sex_id = h.id
                left join workclasses as i ON a.workclass_id = i.id")



####################################################
## 2. Export the "flattened" table to a CSV file. ##
####################################################

write.csv(flat, "C:/Users/Ben/Documents/GitHub/exercises/exercise01/Flat_Census_96.csv", row.names=FALSE)


##################################################################################
## 3. Import the "flattened" table (or CSV file) into your open source analytic ##
## environment of choice (R, Python, Java, etc.) and stage it for analysis.     ##
##################################################################################
remove(list=ls())
final.data <- read.csv("C:/Users/Ben/Documents/GitHub/exercises/exercise01/Flat_Census_96.csv", header = TRUE)


#######################################################################################################################
## 4. Perform some simple exploratory analysis and generate summary statistics to get a sense of what is in the data.##
#######################################################################################################################

## take a quick look at the the contents of this data ##
## continuous vars first ##
summary(final.data[, c("age", "education_num", "capital_gain", "capital_loss", "hours_week")])

## take a look at the distribution of age and also break out by $50k+ or not ##
hist(final.data$age, xlab = "Age")
hist(final.data$hours_week, xlab = "Hours/week")
nrow(final.data[final.data$hours_week == 40,])/nrow(final.data)  ## 46.687% of people work 40 hours a week

## what about the proportion at each age that are making over $50k? ##
ggplot(final.data, aes(x = age, fill = as.factor(over_50k))) + geom_bar(position = "fill", binwidth = 5)

## what about distribution of education level?? ##
hist(final.data$education_num, xlab="Education")
ggplot(final.data, aes(x = education_num, fill = as.factor(over_50k))) + geom_bar(position = "fill") + ylab("Proportion Making $50k+") + xlab(label = "Education Level") + scale_fill_manual(values=c("#E69F00", "#56B4E9") ,name="Salary Level", labels=c("Less than $50k", "More than $50k"))


## look at the categorical vars now ##
countries <-as.data.frame.matrix(table(final.data$countries, final.data$over_50k))
countries$prcnt_50k_plus <- countries[,2]/(countries[,1]+countries[, 2])

edu_levels <-as.data.frame.matrix(table(final.data$education_levels, final.data$over_50k))
edu_levels$prcnt_50k_plus <- edu_levels[,2]/(edu_levels[,1]+edu_levels[, 2])

marital_status <-as.data.frame.matrix(table(final.data$marital_status, final.data$over_50k))
marital_status$prcnt_50k_plus <- marital_status[,2]/(marital_status[,1]+marital_status[, 2])

occupations <-as.data.frame.matrix(table(final.data$occupations, final.data$over_50k))
occupations$prcnt_50k_plus <- occupations[,2]/(occupations[,1]+occupations[, 2])

races <-as.data.frame.matrix(table(final.data$races, final.data$over_50k))
races$prcnt_50k_plus <- races[,2]/(races[,1]+races[, 2])

relationships <-as.data.frame.matrix(table(final.data$relationships, final.data$over_50k))
relationships$prcnt_50k_plus <- relationships[,2]/(relationships[,1]+relationships[, 2])

sex <-as.data.frame.matrix(table(final.data$sex, final.data$over_50k))
sex$prcnt_50k_plus <- sex[,2]/(sex[,1]+sex[, 2])

workclasses <-as.data.frame.matrix(table(final.data$workclasses, final.data$over_50k))
workclasses$prcnt_50k_plus <- workclasses[,2]/(workclasses[,1]+workclasses[, 2])


######################################################################
## 5. Split the data into training, validation, and test data sets. ##
######################################################################

nrow(final.data)
## 48,842 total records in the data
## put 60% in training set, 20% in validate, and remaining 10% in test

train.rows <- sample(1:48842, 29305)
train.data <- final.data[train.rows,]

remaining <- final.data[-train.rows, ]
row.names(remaining) <- 1:nrow(remaining)
nrow(remaining) ## 19,573 records (2/3 in the validate and 1/3 in the test set)

val.rows <- sample(1:19573, 13049)
validate.data <- remaining[val.rows,]
test.data <- remaining[-val.rows,]

## now we've got three datasets: train.data, validate.data, and test.data ##
## model away! ##


#########################################################################################
## 6. Develop a model that predicts whether individuals, based on the census variables ##
## provided, make over $50,000/year. Use `over_50k` as the target variable.            ##
#########################################################################################


## see what the result of stepwise regression techniques yields (forward, backwards, and both) ##
## first create the model with all of the variables in it ##
Logit.Model.Full <- glm(train.data$over_50k ~ train.data$age + train.data$workclasses + train.data$education_num + train.data$capital_gain + 
                          train.data$capital_loss + train.data$hours_week + train.data$countries + train.data$education_levels +
                          train.data$marital_status + train.data$occupations + train.data$races + train.data$relationships + train.data$sex, family=binomial(logit))

## ruh roh... there is perfect seperation on one of the variables used in the model above, I believe it is in the COUNTRIES variable ##
## after looking at the tables created above, there is complete separation in the COUNTRIES (holland) and WORKCLASSES (never worked) ##
## both of these variables appear to provide some separation in % that make more than $50k, so I plan on grouping them ##

## remove the factor levels
temp <- data.frame(lapply(train.data[, 16:length(names(train.data))], as.character), stringsAsFactors = FALSE)
train.data <- cbind(train.data[, 1:15], temp)

## re-group the Countries variable
countries[order(countries$prcnt_50k_plus),]
train.data$countries.grouped <- ifelse(train.data$countries %in% row.names(countries[countries$prcnt_50k_plus > 0.2,]), "Higher_salaries", "Lower_salaries")

## regroup the WORKCLASSES variable
workclasses[order(workclasses$prcnt_50k_plus),]
## split out the NEVER-Worked, missing, and without pay folks in one group -- keep others in their own group ##
train.data$workclasses.grouped <- ifelse(train.data$workclasses %in% row.names(workclasses[workclasses$prcnt_50k_plus > 0.2,]), train.data$workclasses, "Never&Missing&NoPay")

## regroup the education variable to simplify -- those with college vs. those without are distinctly different ##
edu_levels[order(edu_levels$prcnt_50k_plus),]
## everyone below HS grad should be grouped together, Doc and Prof School group together, Associates Degree group together
train.data$edu.levels.grouped <- ifelse(train.data$education_levels %in% row.names(edu_levels[edu_levels$prcnt_50k_plus < .10,]), "Less Than HS", 
                                    ifelse(train.data$education_levels %in% c("Assoc-voc", "Assoc-acdm"), "Associates Deg.",
                                           ifelse(train.data$education_levels %in% c("Doctorate", "Prof-school"), "Doc or Prof", train.data$education_levels)))

## regroup capital gains and losses, should be sufficient to break into two groups each: ZERO or NONZERO ##
train.data$cap.loss.grouped <- ifelse(train.data$capital_loss == 0, "ZERO", "NONZERO")
train.data$cap.gain.grouped <- ifelse(train.data$capital_gain == 0, "ZERO", "NONZERO")


## regroup the AGE variable -- make it a categorical binned variable ##
train.data$age.grouped <- ifelse(train.data$age <=25, "LTE_25",
                                 ifelse(train.data$age >25 & train.data$age <= 37.5, "25_37.5",
                                        ifelse(train.data$age > 37.5 & train.data$age <=50, "37.5_50",
                                               ifelse(train.data$age > 50 & train.data$age <= 62.5, "50_62.5",
                                                  ifelse(train.data$age > 62.5 & train.data$age <= 75, "62.6_75", "75+")))))


## next create the model mith nothing in it (as starting point for FORWARDS stepwise)
Logit.Model <- glm(train.data$over_50k ~ 1, family=binomial(logit))

forward.model <- step(Logit.Model, ~ train.data$age.grouped + train.data$workclasses.grouped + train.data$cap.gain.grouped + 
                  train.data$cap.loss.grouped + train.data$hours_week + train.data$countries.grouped + train.data$edu.levels.grouped +
                  train.data$marital_status + train.data$occupations + train.data$races + train.data$relationships + train.data$sex, direction="forward")
summary(forward.model)

## "occupations" variable doesn't seem to be giving us much... nor does relationships, or races  -- play around with taking the out ##

## run a BACKWARDS stepwise procedure using a full model that has the new grouped vars ##
Logit.Model.Full <- glm(over_50k ~ age.grouped + workclasses.grouped + cap.gain.grouped + 
                          cap.loss.grouped + hours_week + countries.grouped + edu.levels.grouped +
                          marital_status + occupations + races + relationships + sex, data=train.data, family=binomial(logit))

summary(Logit.Model.Full)
## our forward model ended up using everything we gave it, matches the FULL MODEL

backward.model <- step(Logit.Model.Full, direction="backward")
## no vars removed, kept them all in there matching the FULL MODEL


## take a look at the odds ratios ##
OR <- exp(coef(Logit.Model.Full)[-1])
OR
OR.CI <- exp(cbind(OR = coef(Logit.Model.Full), confint(Logit.Model.Full)))[-1,]
OR.CI



## take a look at the ROC curve
Model.ROC <- roc(Logit.Model.Full$y, Logit.Model.Full$fitted)
print(Model.ROC)
plot(Model.ROC)

Class.Table <- cbind(Model.ROC$thresholds, Model.ROC$sensitivities, Model.ROC$specificities)
colnames(Class.Table) <- c("Probability", "Sensitivity", "Specificity")
## find the cutoff prob where we maximize the sum of Sensitivity and Specificity
Class.Table <- cbind(Class.Table, Class.Table[,"Sensitivity"] + Class.Table[,"Specificity"])
ordered <- as.data.frame(Class.Table)
ordered <- ordered[order(ordered$V4),]
tail(ordered)

## Looks as though we get our best predictions with a cutoff point of .2136357

################################################################################
## 6.a  Validate the model -- make any final last adjustments before testing  ##
################################################################################

## first need to transform the validation data set the same way that the training data was altered (group variables)
## remove the factor levels
temp <- data.frame(lapply(validate.data[, 16:length(names(validate.data))], as.character), stringsAsFactors = FALSE)
validate.data <- cbind(validate.data[, 1:15], temp)

## re-group the Countries variable
countries[order(countries$prcnt_50k_plus),]
validate.data$countries.grouped <- ifelse(validate.data$countries %in% row.names(countries[countries$prcnt_50k_plus > 0.2,]), "Higher_salaries", "Lower_salaries")

## regroup the WORKCLASSES variable
workclasses[order(workclasses$prcnt_50k_plus),]
## split out the NEVER-Worked, missing, and without pay folks in one group -- keep others in their own group ##
validate.data$workclasses.grouped <- ifelse(validate.data$workclasses %in% row.names(workclasses[workclasses$prcnt_50k_plus > 0.2,]), validate.data$workclasses, "Never&Missing&NoPay")

## regroup the education variable to simplify -- those with college vs. those without are distinctly different ##
edu_levels[order(edu_levels$prcnt_50k_plus),]
## everyone below HS grad should be grouped together, Doc and Prof School group together, Associates Degree group together
validate.data$edu.levels.grouped <- ifelse(validate.data$education_levels %in% row.names(edu_levels[edu_levels$prcnt_50k_plus < .10,]), "Less Than HS", 
                                        ifelse(validate.data$education_levels %in% c("Assoc-voc", "Assoc-acdm"), "Associates Deg.",
                                               ifelse(validate.data$education_levels %in% c("Doctorate", "Prof-school"), "Doc or Prof", validate.data$education_levels)))

## regroup capital gains and losses, should be sufficient to break into two groups each: ZERO or NONZERO ##
validate.data$cap.loss.grouped <- ifelse(validate.data$capital_loss == 0, "ZERO", "NONZERO")
validate.data$cap.gain.grouped <- ifelse(validate.data$capital_gain == 0, "ZERO", "NONZERO")

## regroup the AGE variable -- make it a categorical binned variable ##
validate.data$age.grouped <- ifelse(validate.data$age <=25, "LTE_25",
                                 ifelse(validate.data$age >25 & validate.data$age <= 37.5, "25_37.5",
                                        ifelse(validate.data$age > 37.5 & validate.data$age <=50, "37.5_50",
                                               ifelse(validate.data$age > 50 & validate.data$age <= 62.5, "50_62.5",
                                                      ifelse(validate.data$age > 62.5 & validate.data$age <= 75, "62.6_75", "75+")))))


val.predictions <- predict(Logit.Model.Full, validate.data)
val.predictions <- cbind(validate.data, val.predictions)
val.predictions$prob <- (exp(val.predictions$val.predictions)/(1+exp(val.predictions$val.predictions))) ## translate from log-odds to prob
val.predictions$pred <- ifelse(val.predictions$prob >= 0.2136357, 1, 0)  ## use the same cutoff point as we found using the ROC curve above


## how well did we do using our model and that cutoff??
nrow(val.predictions[val.predictions$over_50k == val.predictions$pred,])/nrow(val.predictions) ## 79.6% Accurately Identified
nrow(val.predictions[val.predictions$over_50k == 0 & val.predictions$pred == 1,])/nrow(val.predictions) ## 17.30% were falsely identified as making more than $50k
nrow(val.predictions[val.predictions$over_50k == 1 & val.predictions$pred == 0,])/nrow(val.predictions) ## 3.3% made at least $50k but were not predicted as such



################################################################################
## 6.b  Test the model -- Final scores on how well our model did  ##
################################################################################

## first need to transform the test data set the same way that the training data was altered (group variables)
## remove the factor levels
temp <- data.frame(lapply(test.data[, 16:length(names(test.data))], as.character), stringsAsFactors = FALSE)
test.data <- cbind(test.data[, 1:15], temp)

## re-group the Countries variable
countries[order(countries$prcnt_50k_plus),]
test.data$countries.grouped <- ifelse(test.data$countries %in% row.names(countries[countries$prcnt_50k_plus > 0.2,]), "Higher_salaries", "Lower_salaries")

## regroup the WORKCLASSES variable
workclasses[order(workclasses$prcnt_50k_plus),]
## split out the NEVER-Worked, missing, and without pay folks in one group -- keep others in their own group ##
test.data$workclasses.grouped <- ifelse(test.data$workclasses %in% row.names(workclasses[workclasses$prcnt_50k_plus > 0.2,]), test.data$workclasses, "Never&Missing&NoPay")

## regroup the education variable to simplify -- those with college vs. those without are distinctly different ##
edu_levels[order(edu_levels$prcnt_50k_plus),]
## everyone below HS grad should be grouped together, Doc and Prof School group together, Associates Degree group together
test.data$edu.levels.grouped <- ifelse(test.data$education_levels %in% row.names(edu_levels[edu_levels$prcnt_50k_plus < .10,]), "Less Than HS", 
                                           ifelse(test.data$education_levels %in% c("Assoc-voc", "Assoc-acdm"), "Associates Deg.",
                                                  ifelse(test.data$education_levels %in% c("Doctorate", "Prof-school"), "Doc or Prof", test.data$education_levels)))

## regroup capital gains and losses, should be sufficient to break into two groups each: ZERO or NONZERO ##
test.data$cap.loss.grouped <- ifelse(test.data$capital_loss == 0, "ZERO", "NONZERO")
test.data$cap.gain.grouped <- ifelse(test.data$capital_gain == 0, "ZERO", "NONZERO")


## regroup the AGE variable
test.data$age.grouped <- ifelse(test.data$age <=25, "LTE_25",
                                    ifelse(test.data$age >25 & test.data$age <= 37.5, "25_37.5",
                                           ifelse(test.data$age > 37.5 & test.data$age <=50, "37.5_50",
                                                  ifelse(test.data$age > 50 & test.data$age <= 62.5, "50_62.5",
                                                         ifelse(test.data$age > 62.5 & test.data$age <= 75, "62.6_75", "75+")))))


test.predictions <- predict(Logit.Model.Full, test.data)
test.predictions <- cbind(test.data, test.predictions)
test.predictions$prob <- (exp(test.predictions$test.predictions)/(1+exp(test.predictions$test.predictions))) ## translate from log-odds to prob
test.predictions$pred <- ifelse(test.predictions$prob >= 0.2136357, 1, 0)  ## use the same cutoff point as we found using the ROC curve above


## how well did we do using our model and that cutoff??
nrow(test.predictions[test.predictions$over_50k == test.predictions$pred,])/nrow(test.predictions) ## 78.7% Accurately Identified
nrow(test.predictions[test.predictions$over_50k == 0 & test.predictions$pred == 1,])/nrow(test.predictions) ## 17.9% were falsely identified as making more than $50k
nrow(test.predictions[test.predictions$over_50k == 1 & test.predictions$pred == 0,])/nrow(test.predictions) ## 3.3% made at least $50k but were not predicted as such
nrow(test.predictions[test.predictions$over_50k == 1 & test.predictions$pred == 1,])/nrow(test.predictions[test.predictions$over_50k == 1,]) ## 86.7% made at least $50k but were not predicted as such
nrow(test.predictions[test.predictions$over_50k == 0 & test.predictions$pred == 0,])/nrow(test.predictions[test.predictions$over_50k == 0,]) ## 75.6% made at least $50k but were not predicted as such



##############################################################################################
## 7. Generate a chart that you feel conveys 1 or more important relationships in the data. ##
##############################################################################################

## one of the strongest relationships observed was between Education Level and >$50k ##
p <- ggplot(final.data, aes(x = education_num, fill = as.factor(over_50k))) + geom_bar(position = "fill") + ylab("Proportion of Population") + xlab(label = "Education Level") + scale_fill_manual(values=c("#E69F00", "#56B4E9") ,name="Salary Level", labels=c("Less than $50k", "More than $50k"))
ggsave(filename = "C:/Users/Ben/Documents/GitHub/exercises/exercise01/Edu_vs_Income.png", plot = p, width = 5, height = 3, units = "in")


