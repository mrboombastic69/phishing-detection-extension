import torch
import torch.nn as nn
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import PorterStemmer
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')
stemmer = PorterStemmer()

# Define the GRU model with the provided values
class GRUEmailClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUEmailClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class EmailPhishingDetector:
    def __init__(self, model_path, vectorizer_path, input_dim, hidden_dim=128, output_dim=2, n_layers=2, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, input_dim, hidden_dim, output_dim, n_layers)
        self.vectorizer = joblib.load(vectorizer_path)

    # def preprocess(self, text):
    #     # Provided preprocessing code
    #     doc = nlp(text)
    #     no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    #     token = " ".join(no_stop_words)
    #     token = [token.lemma_ for token in nlp(token)]
    #     token = " ".join(token)
    #     token = [stemmer.stem(word) for word in token.split()]
    #     token = " ".join(token)
    #     return token
    def preprocess(self,text):
            text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove excessive whitespace and newlines, replace with a single space
            text = re.sub(r'\s+', ' ', text)
            
            # Strip leading and trailing whitespace
            text = text.strip()
            
            # Remove any remaining special characters (excluding spaces)
            text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenize and process the text using spaCy
            doc = nlp(text)
            
            # Filter out stop words and punctuation
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
            
            # Lemmatize the tokens
            lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
            
            # Stem the lemmatized tokens
            stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
            
            # Join the stemmed tokens back into a single string
            preprocessed_text = " ".join(stemmed_tokens)
        
            return preprocessed_text
    # Remove newlines, tabs, and excessive whitespace
    def transform_text_to_features(self, text):
        text_tfidf = self.vectorizer.transform([text]).toarray()
        return torch.tensor(text_tfidf, dtype=torch.float32).to(self.device)

    def predict(self, email_content):
        preprocessed_text = self.preprocess(email_content)
        email_tensor = self.transform_text_to_features(preprocessed_text)
        email_tensor = email_tensor.unsqueeze(1)  # Add batch dimension
        with torch.no_grad():
            outputs = self.model(email_tensor)
            _, predicted = torch.max(outputs, 1)
            return "Phishing" if predicted.item() == 1 else "Legitimate"

    def load_model(self, model_path, input_dim, hidden_dim=128, output_dim=2, n_layers=2):
        model = GRUEmailClassifier(input_dim, hidden_dim, output_dim, n_layers).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

# # Example usage
# model_path = 'gru_email_classifier.pth'
# vectorizer_path = 'tfidf_vectorizer.pkl'
# input_dim = 5000  # This should match the number of features used for TF-IDF

# detector = EmailPhishingDetector(model_path, vectorizer_path, input_dim)

# # Example email content
# email_content = "Your email content here"  # Replace with actual email content
# prediction = detector.predict(email_content)
# print(f"Prediction: {prediction}")

# # Save the model (if needed)
# detector.save_model('path_to_save_model.pth')
def is_phishing(email_content):
    model_path = 'gru_email_classifier.pth'
    vectorizer_path = 'tfidf_vectorizer.pkl'
    input_dim = 5000  # This should match the number of features used for TF-IDF

    detector = EmailPhishingDetector(model_path, vectorizer_path, input_dim)
    result = detector.predict(email_content)
    return result == "Phishing"

