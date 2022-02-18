### Project done in winter semester 2021/2022 at Johannes Kepler University in Linz
--------------------------------------------------------------------------------

# Explainable AI Assignment 2 - Model Explanations
In this assignment, you are challenged to explain a model. For this, you will research exisiting approaches and apply them to your model and interpret the results.

## General Information Submission due on 02.12.2021, 23:59

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Team Name:** A-Team

**Group Members**
| Student ID    | First Name  | Last Name      | E-Mail |  Workload [%] |
| --------------|-------------|----------------|--------|---------------|
| k51824612        | Vojtech      | Vlcek            |k51824612@students.jku.at  |1/3          |
| k12123727        | Agata        | Kaczmarek        |k12123727@students.jku.at  |1/3        |
| k51831785        | Simeon      | Quant         |k51831785@students.jku.at  |1/3         |

### Model and Dataset

We chose a random forest classifier as a model which we would like to explain. As a dataset for our model, we decided to use a heart attack dataset which contains patients’ medical information based on which we will try to predict the chance of a patient getting a heart attack. We downloaded the dataset from Kaggle (https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset), from the user Rashik Rahman. The main objective of our project is to clearly communicate and explain predictions of the random forest.

## Intermediate Presentation on the 09.12.2021 (optional)
The presentations will be held during the Q&A sessions on that day.
For more details, please refer to the assignment description.

## Final Submission due on 15.12.2021, 23:59
The submission is done with this repository. Make to push your code until the deadline.

The repository has to include the implementations of the picked approaches and the filled out report in this README.

* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Save your final executed notebook(s) as html (File > Download as > HTML) and add them to your repository.

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_model_explanation_2021-<GROUP_NAME>.git
cd xai_model_explanation_2021-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_model_explanation
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.


## Report

### Model & Data

The model we decided to explain is a Random Forest classifier. The Random Forest is an ensemble method which relies on multitude of decision trees. Even though a decision tree is a white-box model, Random Forest becomes hard to visualize and interpret once it reaches a certain level of complexity. Random Forest is widely used and is one of the elementary machine learning models. That is why we decided to focus our efforts on trying to find methods which would help to make this model class more interpretable. 

The goal of our model is to predict whether a patient is at a high risk of getting a heart attack in the future. We trained our model on a simple heart attack dataset which contains patients’ medical information. We chose this dataset because we believe that medicine is one of the areas where interpretability of a model is crucial. We downloaded the dataset from Kaggle (https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset), from the user Rashik Rahman. During the training of our model, we were not trying to achieve a perfect model, on the contrary, for example, we intentionally left in the dataset redundant features to show that our methods for explaining the model can point out their redundancy. 

Detailed description of the features in the dataset: 
- age: Age of the patient in years 
- sex: Gender of the patient (1 = male; 0 = female) 
- cp: chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic) 
- trtbps - resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestoral in mg/dl
- fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- restecg: resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy)
- thalach: maximum heart rate achieved
- exang: exercise induced angina (1 = yes; 0 = no)
- oldpeak: ST depression induced by exercise relative to rest
- slp: the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)
- caa: number of major vessels (0-3) colored by flourosopy
- thall: 2 = normal; 1 = fixed defect; 3 = reversable defect; result of thallium stress test
- target: 0 = less chance of heart attack 1 = more chance of heart attack

### Explainability Approaches

#### 1. Global Feature Importance

##### Five W's and One H: Summarization of the Global Feature Importance

1.  **Why** should one use the method?

	Global feature importance is used to determine which features the model considers important and which not, depending on the model and the data it might be necessary to get additional information to use this method appropriately.
    <br>
	
2.  **What** could be visualized?

	The method visualizes the importance of individual features.
	<br>

3. **When** is this method used?

	Global feature importance can be used to find unimportant features after training for simplification of the model.
	<br>

4.  **Who** would benefit from this method?

	 The developers may benefit by being able to simpify their model and being able to check whether the model aligns with their domain knowledge.
	<br>

5.  **How** can this method explain the model?

	Models can be explained globally only. It is a superficial metric.
	<br>

6.  **Where** could this method be used?

	Anywhere, but the metric differs depending on the model.

![global_feature_importance.png](/images/global_feature_importance.png)
	<br> *Figure 1.1 Visualisation ofpermutation feature importance an impurity based feature importance (bars).*

In the plot above the overall feature importance of each feature in the model is shown. The plot is sorted by the mean permutation importance, additionally the impurity based feature importance of the random forest model is shown (as rescaled barchart).

These are two ways to visualize which features the model deems important and which not.
Permutation importance sometimes believes a feature to be unimportant or less important then it is if there is correlation with other features. Therefore, correlation needs to be checked to determine whether features are really unimportant.
Impurity based feature importance normally has a bias towards features with a huge numerical range in contrast to features with a small number of possible values. 

Due to the weaknesses described above this graph cannot be used to make any kind of definitive statement about the model on its own. However, it can be used to gain a feeling about what the model deems important and if applied to a model without correlation between features a statement about the model could be made (at this point we have no information about the correlation between features).

![corr_matrix.png](/images/corr_matrix.png)
	<br> *Figure 1.2 Correlation matrix*
	
There do not seem to be any strong correlations between features, therefore, the permutation importance should be reliable.

#### 2. SHAP Values

SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions. One goal of SHAP is alignment with human intuition.

##### Five W's and One H: Summarization of the SHAP Values

1.  **Why** should one use the method?

	SHAP enables the user to see the influence of features on the models decisions locally. Based on this, one can come up with verbal explanations of the model's decisions and thus increases the trust in the decisions.
	<br>
	
2.  **What** could be visualized?

	Local (sample-wise) feature importance and global feature importance. Here it is also experimentally used to directly visualize trustworthiness.  
	<br>

3. **When** is this method used?

	SHAP is used after training and during application.
	<br>

4.  **Who** would benefit from this method?

	Both the developers of the model and its users can benefit from this method. For use normally domain knowledge is required. The direct visualisation of trustworthiness may make it possible for non-expert users to benefit as well.
	<br>

5.  **How** can this method explain the model?

	It explains the decisions which the models make. This can be used to gain local and global insights.
	<br>

6.  **Where** could this method be used?

	On any model, since SHAP values are an explanation unification approach.


Below there is an example, in which the model came to the conclusion that the patient has a heart disease with 70% certainty:

![single_SHAP.png](/images/single_SHAP.png)
	<br> *Figure 2.1 SHAP values of a single sample*

The most important features which contributed to the decision are:
- caa : number of major vessels (0-3) colored by flourosopy 
- cp : the type of chest pain
- thalachh : the maximum heart rate
- thall : result in thallium stress test

With sufficient knowledge of the domain (for example: as a doctor) one can now deduct whether the decision made was a reasonable decision or not.

If the decision, or more specifically - the features, which the decision was based on, seem unreasonable the decision has to be made manually. If the final decision is the same as that of the model one could try to figure out why there is a correlation and if there is a valid reasoning that can be done or if it is just an error in the model for example due to over-fitting.

Below there are the SHAP values of the whole dataset which was used:

![complete_SHAP.png](/images/complete_SHAP.png)
	<br> *Figure 2.2 SHAP values of the whole test-set*

Now let's try to extract information about the model's decisions in them more global manner. The plot below shows the SHAP values grouped by feature instead of sample:

![beechart_SHAP.png](/images/beechart_SHAP.png)
	<br> *Figure 2.3 SHAP values grouped by feature*

More blue dots are lower values, while more red dots are higher values. The position on the X axis shows the impact of the feature on the decision in its respecive context. The values are sorted by their overall influence on the decision of the model (in regard to this dataset).

The strongest indicator for the decision (on avarage) seems to be 'cp', which is the type of chest pain the patient has. 'cp' can take four different values, three of which indicate heart disease, the last one (most likely no pain) indicates no heart disease.
In a similar manner all features can be said to have low values indicating one tendency and high values indicating the other, this is more clear for discrete values than for values with huge ranges.
Age seems to be an exception to some extent because both low values and high values can be found on the same side of the plot.

##### The Trust Mashine (Experimental)

The method is intended to give non-expert model users confidence about the model's decision by the fact that there are many similar cases which also were classified correctly like this.

Below the SHAP values of the dataset on our model were taken and projected using t-SNE. The hypothesis is, that values which are not near the decision border and in dense clusters will probably be safe decisions which can be trusted.

![projection_SHAP.png](/images/projection_SHAP.png)
	<br> *Figure 2.4 projection of SHAP values*

Next we look for clusters in the projection:

![borders_SHAP.png](/images/borders_SHAP.png)
	<br> *Figure 2.5 projection of SHAP values with decision border and clusters*

The red line is our guess for the decision border. The blue clusters seem trustworthy acording to our hypothesis. The violet clusters might be real but they might also be artefacts or too close to the decision border, since or model only has an accuracy of 88% and mainly false-positives (unhealthy=positive; this is visible in the check below).

Lastly we check whether our hypothesis is correct by checking the true labels:

![check_SHAP.png](/images/check_SHAP.png)
	<br> *Figure 2.6 projection of SHAP values with wrong labels shown to check the hypothesis*

The blue clusters turned out to be valid. However, the lower the accuracy of the model and the bigger the dataset you are working with the more often there will be cases where a cluster which is identified as "probably safe" contains data points which are wrongly classified. Therefore, you could assign each cluster some kind of certainty value which you evaluate based on a huge dataset. This way the method can be applied to problems where classes are partly overlapping.

#### 3. LIME (Local Interpretable Model-Agnostic Explanations)

The main idea of this method is to approximate complicated black-box model by a simpler glass-box one. Usually used with problems having very large number of explanatory variables. In that case the simpler glass-box model is easier to interpret. The main idea is to train a simpler glass-box model on artificial data so that it approximates the predictions of a complicated black-box model (on these data). We decided to use this method in our problem, as medical data usually are complicated and contain a lot of various variables, which are important for explanations. 
We learned about this method in [Molnar, 2021. Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) book and in [Biecek and Burzykowski, 2020. Explanatory Model Analysis](https://ema.drwhy.ai) book.

##### Five W's and One H: Summarization of LIME Method

1.  **Why** should one use the method?

	In this method we create simpler glass-box, which is implemented based on the black-box model and is easier to interpret.
	<br>
	
2.  **What** could be visualized?

	Probabilities of the most important (for decisions) features and their influence (positive or negative, how big) on the model prediction. 
	<br>

3. **When** is this method used?

	It can be used not only after training to evaluate our model, but also during training. For example if we see that some of the features are never important for training, we might consider trying to reduce number of features and seeing, if it changed the accuracy (or other method of evaluation model prediction). Here we used it in both cases.
	<br>

4.  **Who** would benefit from this method?

	Model developers, as implemented method would make their work easier and faster, also more understandable.
	<br>

5.  **How** can this method explain the model?

	The simpler glass-box model is trained on artificial data in that way, that its predictions are similar to the predictions of original model.
	<br>

6.  **Where** could this method be used?

	It could be used for lots of models, one of them is random forest. Also it is usually used to help explain problems, which have a lot of various features. 

Below there are examples of the local explanations made by LIME Method for our model.

![lime_row_4.png](/images/lime_row_4.png)
	<br> *Figure 3.1 projection of LIME values for row 4*
	
![lime_row_7.png](/images/lime_row_7.png)
	<br> *Figure 3.2 projection of LIME values for row 7*
	
![lime_row_8.png](/images/lime_row_8.png)
	<br> *Figure 3.3 projection of LIME values for row 8*

Above shown three results are examples of different probabilities and certainty of the model. In the row 7 & 8 our model was really sure, that the prediction made is correct. So it has nearly no problems in saying, if provided person has high risk of having heart attack or not. Most (or in 8 row all) features are thought to have values which are typical for predicted class. In the row number 4 the situation is much more different. There the model is only 55% sure of its decision. Some of the values of the features also seems to be typical for ill persons. At the end, here the difference is not that big, it seems more like luck if the model chooses correctly.

Based on these results we decided to check which features generally (for all test data) are considered by method as the ones which usually influence the model the most. We try to answer questions: what was the most important feature when predicted illness and what if healthy?

![lime_summary_healthy_one.png](/images/lime_summary_healthy_one.png)
	<br> *Figure 3.4 Summary for most important feature for healthy people*
	
![lime_summary_ill_one.png](/images/lime_summary_ill_one.png)
	<br> *Figure 3.5 Summary for most important feature for ill people*
	
As we see, the values of the features considered are mostly common in a group, at least for ill people. Most of them had less or equal to 0 number of major vessels, which means they had 0, because the values are between 0 and 3. The name of the feature seems to say us, why it was the most important feature. For people considered as more healthy, there was not such a big agreement, which feature helps the most with predictions. Some of the values of 'oldpeak', 'caa' and 'cp' were considered as the most commonly chosen as the most important.

However, the amount of data in our set is not that big and because of that we decided to look on the summary of the most used but first three features for every group.

![lime_summary_healthy_three.png](/images/lime_summary_healthy_three.png)
	<br> *Figure 3.6 Summary for three most important features for healthy people*
	
![lime_summary_ill_three.png](/images/lime_summary_ill_three.png)
	<br> *Figure 3.7 Summary for three most important features for ill people*

Here also for ill people number of major vesses equals to 0 seems to be the most important factor, but also apeared that small values of 'thall' are one of the main factors. Also for healthier people 'oldpeak' and 'caa' seem to be the most useful features while carrying out a classification. What is interesting and for sure not intuitive, is that in these results we see nothing about age of the patient. We decided to check it, training a model without this feature. After checking the results of that model, we can see that the accuracy did not change. Also not many labels are different - only three out of ninty one changed. It means that variable 'age' does not contain much new informations relevant to the model, which other features do not contain. We believe that the older the person, the statistically worster its health, and the values of other features show this. That is why the results do not change much when we delete this feature.
LIME helps to explain and understand the model locally, but can also be helpful with checking which features are considered as more important, and which seems not to be useful.

#### 4. Explainable Matrix

The concept of Explainable Matrix for Random Forest interpretability was first introduced by Neto and Paulovich (2020). This visualization method uses a matrix representation to visualize large quantities of Random Forest's decision rules. Using this method, one can create both local and global explanations and thus analyze either an entire model or audit concrete classification results. 

##### Five W's and One H: Summarization of the Explainable Matrix

1.  **Why** should one use the method?

	Explainable Matrix provides a complete overview of Random Forest's decision rules. Based on these rules, one can audit a model as a whole or its individual decisions for specific instances. The possibility to closely inspect individual decision rules helps the users to understand their models better and thus increases their trust in their decisions.
	<br>
	
2.  **What** could be visualized?

	The method visualizes individual logic rules of the Random Forest using a matrix visualization. It produces both global and local explanations. Furthermore, it visualizes the importance of individual features.
	<br>

3. **When** is this method used?

	Explainable Matrix can be used to inspect a model and its behavior after training. This can lead to removal of some redundant features or simplification of the forest. Moreover, Explainable Matrix is also suitable for use during the usage of a model. Its local explanation is ideal for analyzing which rules had an influence on a certain decision.
	<br>

4.  **Who** would benefit from this method?

	Both the developers of the model and its users can benefit from this method. The developers may benefit from the possibility to inspect the quality of their model using a clearly interpretable visualization. The users may benefit from the local explanations which provide a clear overview of influencers in the model’s prediction making process.
	<br>

5.  **How** can this method explain the model?

	Models can be explained both locally and globally. Explainable Matrix provides an overview of all decision rules of Random Forest, and also an overview of logical rules which led to a classification of individual sample.
	<br>

6.  **Where** could this method be used?

	Explainable Matrix can be used everywhere where there is a need to increase trust in a model by auditing its predictions and the model as a whole.

##### Explanation of the Explainable Matrix

Explainable Matrix explains Random Forest using a matrix visualization, where rows represent decision rules, columns represent features, and cells are rules predicates. Presented decision rules are extracted from decision paths of individual trees in a forest. The rule extraction process is visualized below.

![rule_extraction.png](/images/rule_extraction.png)
	<br> *Figure 4.1. Visualization of the rule extraction process (Neto and Paulovich, 2020).*

As mentioned above, the Explainable Matrix produces both global and local model explanations. Global explanation provides an overview of the whole model, local explanation explains classification of one specific sample. The two figures below visualize a global and local explanation of a small forest consisting of 2 trees trained on our data. 

![small_ex_heart_glob.png](/images/small_example_heart_glob.png)
	<br> *Figure 4.2. Explainable Matrix - global explanation of a small forest. Rules are ordered based on their coverage, features are ordered based on their importance.*


![small_ex_heart_loc_used.png](/images/small_example_heart_loc.png)
	<br> *Figure 4.3. Explainable Matrix - local explanation for sample number 33 going through a small forest. Rules are ordered based on their coverage, features are ordered based on their importance.*

Each row in a matrix represents a decision rule. Certainty of a rule is a vector of probabilities for each class obtained from the leaf node of a decision path. A rule’s class is the class with the highest probability in the leaf node. Lastly, the rule coverage is the number of samples in the training set of the rule’s class for which the rule is valid divided by the total number of samples of the rule’s class in the training set. 

Each column in the matrix represents a feature. For each feature, feature importance is calculate using the Mean Decrease Impurity. 

Cells in the matrix represent rules predicates. They are visualized by a rectangle colored according to the rule’s class. This rectangle is positioned and sized inside a cell proportional to the predicate limits, where a left side of a cell represents minimum value of a feature in the dataset and the right side represents maximum. If no predicate is present, the matrix cell is blank.

![cells_exp.png](/images/cells_exp.png)<br>
	<br> *Figure 4.4. A closer look at the cells in the Explainable Matrix*

The local explanation differs from the global one by visualizing only the rules that were used by the model to classify one concrete sample. Moreover, the visualization also contains additional column “committee’s cumulative voting”. Similarly to the rule certainty, the committee’s cumulative voting is a vector of probabilities for each class considering only the first i rules (based on the matrix order). Lastly, the dashed line in each column in the visualization represents feature values of that specific sample.

##### Explanation of the Model

###### Global Explanation

The goal of global explanation is to provide description of the whole Random Forest based on its decision rules. Figure 4.5 contains the global explanation of our forest. The forest contains in total 1543 rules which rely on 13 features. In the matrix, the rules are ordered based on the rule coverage and features based on their importance. At the first glance, one can clearly see that features resting electrocardiographic results (restecg) and fasting blood sugar (fbs) are used only in few instances and thus have a low importance. Removing these features may prove to be beneficial for the simplicity and overall performance of the forest. By looking closely at individual features and rules, patters in the predicate ranges emerge. These patterns become more pronounced once the focus is on rules with higher coverage. Figure 4.6 provides the same view as figure 4.5 but only with the rules with coverage greater than 0.15. For the most important feature, maximum heart rate achieved (thalach), the predicate ranges indicate that patients with higher values tend to be classified as being at higher risk of getting a heart attack. On the other hand, for example, higher risk patients tend to have lower oldpeak value then the other low risk patients.

![ex_heart_glob.png](/images/our_heart_glob.png)
	<br> *Figure 4.5. Explainable Matrix - global explanation for the Random Forest. Rules are ordered based on their coverage, features are ordered based on their importance.*
	
	
![ex_heart_glob.png](/images/our_heart_glob_filter.png)
	<br> *Figure 4.6. Explainable Matrix - global explanation for the Random Forest. Only rules with coverage greater than 0.15 are present. Rules are ordered based on their coverage, features are ordered based on their importance.*

###### Local Explanation

In a hypothetical scenario, a doctor is informed that his patient, based on his current medical records, may be at a high risk of getting a heart attack. After looking at the patient’s records, the doctor is not fully convinced with the model’s decision and decides to contact the technical support to ask them to audit the model’s decision. Technician from the technical support team creates a local explanation using the Explainable Matrix (figure 4.7) for this specific patient and inspects the decision rules which led to the conclusion that this patient is at a high risk. Even though the patient's values for the two most important features (thalach, oldpeak) are often at the borders of the predicate ranges, majority of the rules with high coverage classify the patient as being at risk with complete certainty. This decision is contradicted by some more specialized rules; however, their rule certainty is often very low. Looking at these results, the technician is confident with the model’s decision and informs the doctor.

![ex_heart_loc_used.png](/images/our_heart_loc.png)
	<br> *Figure 4.7. Explainable Matrix - local explanation for sample number 33. Rules are ordered based on their coverage, features are ordered based on their importance.*


### Summary of Approaches

Above we described four different approaches to explain our model and the most important features according to these methods. Some of the approaches were more complicated to implement and adjust for our data (as Explainable Matrix) and with some were easier to use and get the results (LIME, Feature Importance). We are not doctors and have not a deep knowledge about medical data, therefore we cannot have big intuition how the results of explanations should look like. That is why for more specified problems, more difficult data as medical ones, Machine Learning and Explainable AI experts are not the only ones to create good model for predictions and find the best explaination techniques for it. 

However, we could try to compare our explanations and actually we managed to do it and to find some of the similarities between them. In our project we used both LIME and SHAP methods, which seems to be similar, so at first we would like to compare their results. Both methods were used in local and global explanations. From the results of both of them we were able to see, that there is something unintuitive with 'age' feature, before we thought it should have much more impact on the result. Both methods predicted 'caa' and 'cp' features as one of the most important, but LIME additionaly predicted 'oldpeak', whereas SHAP - 'thalachh' and 'thall'. Comparing to other methods - Feature Importance and Explainable Matrix - all of these features were one of the most important (except from 'thall'). For the other methods it was easier to see the most important features, as they are more intended to global explanations than LIME and SHAP. Overall, we think that our explanations are accurate and we would believe our model in predicting our health condition.

### References

1. Biecek and Burzykowski, 2020. Explanatory Model Analysis
2. Lundberg, Scott & Lee, Su-In. (2017). A Unified Approach to Interpreting Model Predictions.
3. Molnar, 2021. Interpretable Machine Learning
4. Neto, M.P. and Paulovich, F.V., 2020. Explainable Matrix-Visualization for Global and Local Interpretability of Random Forest Classification Ensembles. IEEE Transactions on Visualization and Computer Graphics, 27(2), pp.1427-1437.

