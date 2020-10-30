def model_predict_proba(data):
	import pandas as pd
	from joblib import load

	#from model building
	model = load("ML/utils/joblibs/model.joblib")
	selected_features = load("ML/utils/joblibs/selected_features.joblib")
	features_type = load("ML/utils/joblibs/features_type.joblib")
	encoders = load("ML/utils/joblibs/encoders.joblib")

	#data: list -> DataFrame
	d = {}
	for i in range(0, len(selected_features)):
		if features_type[i] == 'numeric':
			d[selected_features[i]] = [data[i]]
		elif features_type[i] == 'cat':
			d[selected_features[i]] = [encoders[selected_features[i]].transform([data[i]])]
	df = pd.DataFrame(data=d, columns=selected_features)

	#model
	proba = model.predict_proba(df)

	return proba[0]