from flask import Flask, request, jsonify

from database.db import add_or_update_email_data, add_or_update_url_data
from detection import model_class

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
    return jsonify(response)

@app.route('/check/url', methods=['POST'])
def check_url():
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'success': False, 'message': 'Missing URL'}), 400

    # Call your ML model function here
    is_phishing = is_phishing_url(url)
    response = {'success': True, 'is_phishing': is_phishing}
    return jsonify(response)

def is_phishing_email(email_content, sender):
    ...
    return True

def is_phishing_url(url):
    return model_class.is_phishing(url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
