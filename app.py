from flask import Flask, render_template, request, jsonify

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

app = Flask(__name__)

# Train the model and initialize CountVectorizer only once
data = pd.read_csv("data/Youtube01-Psy.csv")
data = data[["CONTENT", "CLASS"]]
data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
x = cv.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

model = BernoulliNB()
model.fit(xtrain, ytrain)

def predict_spam(comment):
    data = cv.transform([comment]).toarray()
    return model.predict(data)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_spam', methods=['POST'])
def detect_spam():
    if request.method == 'POST':
        comment = request.form['comment']
        prediction = predict_spam(comment)
        result = {"comment": comment, "prediction": prediction}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
