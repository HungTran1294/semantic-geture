from flask import Flask, render_template, request
import pickle
from model import preprocess_text

app = Flask(__name__)

# Load model và vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review = request.form['review']
        # Tiền xử lý, chuyển đổi review thành vector và dự đoán
        preprocessed_review = preprocess_text(review)
        review_vec = vectorizer.transform([preprocessed_review])
        prediction = model.predict(review_vec)[0]
        sentiment = "Tích cực" if prediction == 1 else "Tiêu cực"
        return render_template('index.html', review=review, sentiment=sentiment)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)