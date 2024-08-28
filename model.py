import nltk
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download dataset (nếu cần)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Đọc dữ liệu mẫu (bạn có thể thay thế bằng dữ liệu thực tế)
# reviews = [
#     ("Bộ phim thật tuyệt vời! Tôi rất thích nó.", 1),
#     ("Diễn xuất quá tệ. Kịch bản nhàm chán.", 0),
#     ("Một bộ phim đáng xem. Tôi rất xúc động.", 1),
    # ... thêm dữ liệu vào đây ...
# ]
df = pd.read_csv('IMDB Dataset.csv')

X = df['review'].values  # Lấy cột "review" làm input
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
# X = [review[0] for review in reviews]
# y = [review[1] for review in reviews]


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))  # Sử dụng stop words tiếng Anh

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    return " ".join(words)

X = [preprocess_text(review) for review in X]

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo vectorizer (TF-IDF) và huấn luyện model Logistic Regression
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Chuyển đổi dữ liệu kiểm tra bằng vectorizer đã được fit

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_vec)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Lưu model và vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)