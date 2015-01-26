## RTI CDS Analytics Exercise 01 Response

Below is a written description of the methodology and results of a predictive modeling exercise given by RTI CDS 

------

### The Problem

Using 1996 US Census data on nearly 50,000 individuals that contains information including their age, sex, level of education, and work experience, 
a logistic regression model was built to predict whether or not an individual makes more than $50,000. Analysis was done in order to determine
which of these available factors could aid in determining the level of an individual's income. Some of the observations made during this exploratory
analysis include:
- The age for observed individuals ranges from 17 to 90 years old, with an inner-quartile-range of 28 to 48
- Individuals work anywhere from 0 to 99 hours a week, but a large portion work 40 hours/week (47 percent)
- A large gender disparity exists: more than 30 percent of males make more than $50,000 but just 11 percent of females do
- Those that are married and living with their spouse are far more likely to make more than $50,000 than others that are any of the following: divorced, spouse-absent, never-married, separated, or widowed.

One of the strongest relationships observed was that between the level of education that an individual had and their level of income. The chart below shows the proportion of
individuals at every education level (one through sixteen) that are making more than $50,000. As seen below, the greater one's education level the more likely they are to make
at least $50,000. The largest jump appears between those who have not completed high school and those that have, with those on opposite sides of the spectrum having stark differences.
While less than one-fourth of the individuals without a high school diploma make more than $50,000, nearly three-fourths of those with doctorates or professional degrees do.

<img src="https://raw.githubusercontent.com/bheal521/exercises/master/exercise01/Edu_vs_Income.png" alt="Education-Level" width="100%", height="100%">


### The Model

When designing the logistic regression model, stepwise regression techniques were used to explore several baseline models. Through this process, quasi-complete separation was encountered
due to several combinations of the levels of input variables all having the same level of income. In order to simplify the design space, efforts were made to regroup several of levels of the categorical
variables throughout to be useful for prediction. Re-grouping the levels within these variable solved issued with quasi-complete separation. The following variables were re-binned during this process:
- `Work classes`: Individuals that had occupation information missing, or had never worked, or were without pay were grouped together.
- `Education Level`: Individuals with less than a high school degree were grouped together, individuals with associates degrees were grouped together, and individuals with doctorates or professional degrees were grouped together.
- `Capital Gains`: All individuals with zero capital gains were grouped together, and all those with non-zero capital gains were grouped together.
- `Capital Loss`: All individuals with zero capital loss were grouped together and all those with non-zero capital loss were grouped together.
- `Age`: Individuals' ages were binned as those less than 25, 25 to 37, 38 to 50, 51 to 62, 63 to 75, and over 75. 

Once binned, these factors were used as input variables to the logistic model. Additionally, an individuals marital status, occupation, race, relationship, gender and number of hours per week worked were used
to predict whether or not an individual had an income of more than $50,000. Once a model was created, the accuracy of the model was observed at varying probability cut off points using the ROC curve. The sum of
the sensitivity and specificity were maximized at a cut off probability of approximately 0.21. This cut off level was used for prediction when assessing the accuracy of the model.


### The Results

The logistic regression model explained above was more than 85 percent accurate in identifying individuals with an income of more than $50,000. It was less accurate in its ability to predict individuals
with incomes of less than $50,000 (77 percent). The table below illustrates this classification as was seen when testing the model on a portion of the data that was set aside in order to accurately test
the performance of the model. The cells where text is highlighted indicate records where the model correctly predicted the individual's income level.

| Actual Income | Predicted: $50k + | Predicted: < $50k |
|---------------|:-----------------:|:-----------------:|
| $50k + 	    |***1,266***		| 217				|
| < $50k		|  1,168 			| ***3,859***		|


