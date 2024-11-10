This dataset provides demographic and health data for patients, with features including:

	•	Gender
	•	Age
	•	Medical history (e.g., hypertension, heart disease)
	•	Lifestyle factors (e.g., smoking status)
    •   ...

Using this data, our goal is to build a model that can help medical practitioners assess the probability of a patient having experienced a stroke, given their current and past health profile.

Objectives

	•	Develop a machine learning model to predict if a patient has had a stroke, leveraging available demographic and health data.
	•	Maximize recall to reduce false negatives, prioritizing identification of potential stroke cases.
	•	Deploy a toy version of the model using Docker for demonstration purposes.

Outcome

Through iterative experimentation and classifier comparisons, we determined that logistic regression performed well for this task. Given the data available, we reframed the objective from predicting future strokes to assessing the likelihood of a past stroke. The final logistic regression model was deployed in a Docker container as a prototype.