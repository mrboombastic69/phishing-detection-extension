from flask import Flask, request, jsonify
from db import add_or_update_email_data

import email_phishing

import torch
import torch.nn as nn
import joblib
import pickle
import numpy as np
import re
import whois
from datetime import datetime
import requests
import dns.resolver
import random
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urlsplit
from sklearn.preprocessing import MinMaxScaler

# Define the model class
class ChurnModel(nn.Module):
    def __init__(self):
        super(ChurnModel, self).__init__()
        self.layer_1 = nn.Linear(23, 300)
        self.layer_2 = nn.Linear(300, 100)
        self.layer_out = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(300)
        self.batchnorm2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x

# Define the prediction class
class MLPModel:
    def __init__(self, model_path):
        # Load the model object
        
        #self.model = joblib.load(model_path)
        # Load the model state_dict using pickle
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, input_data):
        # Convert to a 2D tensor (batch size of 1)
        # input_tensor = torch.tensor([input_data], dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():  # No gradient calculation needed for inference
            outputs = self.model(input_data)
        
        # Convert the output to a binary prediction (1 or 0)
        prediction = (outputs.numpy() > 0.5).astype(int)
        return prediction
class UrlAnalysis:
    def __init__(self) -> None:
        # Load the scaler using joblib
        self.scaler = joblib.load('scaler.pkl')    
    def url_features(self, url: str) -> tuple:
        """
        Extract various features from the given URL.
        
        Parameters:
        url (str): The URL to extract features from.
        
        Returns:
        tuple: A tuple containing the values of all extracted features.
        """
        def length_url(url: str) -> int:
            return len(url)

        def length_hostname(url: str) -> int:
            hostname = urlparse(url).hostname
            return len(hostname) if hostname else 0

        def ip(url: str) -> int:
            hostname = urlparse(url).hostname
            ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
            return int(bool(hostname and ip_pattern.match(hostname)))

        def nb_dots(url: str) -> int:
            return url.count('.')

        def nb_qm(url: str) -> int:
            return url.count('?')

        def nb_eq(url: str) -> int:
            return url.count('=')

        def nb_slash(url: str) -> int:
            return url.count('/')

        def nb_www(url: str) -> int:
            return url.lower().count('www')

        def ratio_digits_url(url: str) -> float:
            digits = sum(c.isdigit() for c in url)
            return digits / len(url) if len(url) > 0 else 0

        def ratio_digits_host(url: str) -> float:
            hostname = urlparse(url).hostname
            if hostname:
                digits = sum(c.isdigit() for c in hostname)
                return digits / len(hostname) if len(hostname) > 0 else 0
            return 0

        def tld_in_subdomain(url: str) -> int:
            subdomain = urlparse(url).hostname.split('.')[0]
            tlds = [".com", ".net", ".org", ".info", ".biz", ".io", ".co"]
            return int(any(tld in subdomain.lower() for tld in tlds))

        def prefix_suffix(url: str) -> bool:
            hostname = urlparse(url).hostname
            return int('-' in hostname)

        def shortest_word_host(url) -> int:
            hostname = urlparse(url).hostname
            if hostname:
                words = hostname.split('.')
                shortest_word = min(words, key=len)
                return len(shortest_word)
            return 0

        def longest_words_raw(url) -> int:
            words = url.split('/')
            longest_word = max(words, key=len)
            return len(longest_word)

        def longest_word_path(url) -> int:
            path = urlparse(url).path
            if path:
                words = path.split('/')
                longest_word = max(words, key=len)
                return len(longest_word)
            return 0

        def phish_hints(url: str) -> int:
            phishing_keywords = ['verify', 'account', 'login', 'update', 'banking', 'secure', 'signin', 'ebayisapi']
            return sum(keyword in url.lower() for keyword in phishing_keywords)

        def nb_hyperlinks(url):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a')
                return len(links)
            except requests.RequestException:
                return 0

        def ratio_intHyperlinks(url):
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.find_all('a')
                total_links = len(links)
                if total_links == 0:
                    return 0.0
                domain = urlparse(url).hostname
                internal_links = [link for link in links if domain in link.get('href', '')]
                return len(internal_links) / total_links
            except requests.RequestException:
                return 0.0
        def empty_title(url) -> int:
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                # Check if soup.title is None
                if soup.title is None:
                    return 1  # Title is absent
                # Ensure title.string is safely accessed
                title = soup.title.string if soup.title.string else ''
                return int(len(title.strip()) == 0)
            except requests.RequestException as e:
                print(f"RequestException: {e}")  # Debugging: Print exception message
                return 1
            except Exception as e:
                print(f"Unexpected error: {e}")  # Debugging: Print any unexpected errors
                return 1

        # def domain_in_title(url):
        #     try:
        #         response = requests.get(url, timeout=5)
        #         response.raise_for_status()
        #         soup = BeautifulSoup(response.content, 'html.parser')
        #         title = soup.title.string if soup.title else ''
        #         domain = urlparse(url).hostname
        #         return int(domain in title) if domain else 0
        #     except requests.RequestException:
        #         return 0
        def domain_in_title(url) -> int:
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                # Check if soup.title is None
                if soup.title is None or soup.title.string is None:
                    return 0  # Domain cannot be in title if title is None or empty
                title = soup.title.string
                domain = urlparse(url).hostname
                # Ensure domain is safely checked
                return int(domain in title) if domain else 0
            except requests.RequestException as e:
                print(f"RequestException: {e}")  # Debugging: Print exception message
                return 0
            except Exception as e:
                print(f"Unexpected error: {e}")  # Debugging: Print any unexpected errors
                return 0
        def domain_age(url: str) -> int:
            domain = urlparse(url).netloc
            try:
                whois_info = whois.whois(domain)
                if whois_info.creation_date:
                    now = datetime.now()
                    domain_age = (now - whois_info.creation_date).days
                    return domain_age
                else:
                    return 0
            except:
                return 0

        def google_index(url: str) -> int:
            return 0

        def page_rank(url: str) -> int:
            return 0

        features = (
            length_url(url),
            length_hostname(url),
            ip(url),
            nb_dots(url),
            nb_qm(url),
            nb_eq(url),
            nb_slash(url),
            nb_www(url),
            ratio_digits_url(url),
            ratio_digits_host(url),
            tld_in_subdomain(url),
            prefix_suffix(url),
            shortest_word_host(url),
            longest_words_raw(url),
            longest_word_path(url),
            phish_hints(url),
            nb_hyperlinks(url),
            ratio_intHyperlinks(url),
            empty_title(url),
            domain_in_title(url),
            domain_age(url),
            google_index(url),
            page_rank(url)
        )

        return features

    def scaled_and_tensor_features(self, url: str) -> torch.Tensor:
        features = self.url_features(url)
        features_array = np.array(features).reshape(1, -1)  # Reshape for scaler
        scaled_features = self.scaler.transform(features_array)
        features_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        return features_tensor        
def is_phishing_url(url: str) -> bool:
    # Define the path to the model directly
    model_path = 'mlp_model.pkl'
    # Initialize UrlAnalysis and MLPModel
    url_analysis = UrlAnalysis()
    model = MLPModel(model_path)
    # Extract and preprocess features
    features_tensor = url_analysis.scaled_and_tensor_features(url)
    # Predict using the model
    prediction = model.predict(features_tensor)
    # Return prediction as an integer (0 or 1)
    return bool(prediction[0][0])

app = Flask(__name__)

@app.route('/check/email', methods=['POST'])
def check_email():
    data = request.json
    email_content = data.get('email_content')
    sender = data.get('sender')

    if not email_content or not sender:
        print("Bad Request")
        return jsonify({'success': False, 'message': 'Missing email content or sender'}), 400
    
    # Call your ML model function here
    is_phishing = is_phishing_email(email_content, sender)
    response = {'success': True, 'is_phishing': is_phishing}
    # print({"content": email_content})
    # print(response)
    print({"Email Content": email_content, "is_phishing": is_phishing})

    return jsonify(response)

@app.route('/check/url', methods=['POST'])
def check_url():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'success': False, 'message': 'Missing URL'}), 400

    # Call your ML model function here
    is_phishing = is_phishing_url(url)
    print({"url": url, "is_phishing": is_phishing})
    response = {'success': True, 'is_phishing': is_phishing}
    return jsonify(response)

def is_phishing_email(email_content, sender):
    res = email_phishing.is_phishing(email_content)
    if res == True:
        add_or_update_email_data(sender)
    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
