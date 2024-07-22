import sqlite3

DATABASE_NAME = 'phishing_data.db'

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS email_data (
            sender TEXT PRIMARY KEY,
            phishing_rate INTEGER NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS url_data (
            url TEXT PRIMARY KEY,
            phishing_rate INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_or_update_email_data(sender):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    
    c.execute('SELECT phishing_rate FROM email_data WHERE sender = ?', (sender,))
    row = c.fetchone()
    
    if row:
        # Sender exists, increment phishing_rate
        new_rate = row[0] + 1
        c.execute('UPDATE email_data SET phishing_rate = ? WHERE sender = ?', (new_rate, sender))
    else:
        # Sender does not exist, insert new record
        c.execute('INSERT INTO email_data (sender, phishing_rate) VALUES (?, ?)', (sender, 1))
    
    conn.commit()
    conn.close()

def add_or_update_url_data(url):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    
    c.execute('SELECT phishing_rate FROM url_data WHERE url = ?', (url,))
    row = c.fetchone()
    
    if row:
        # URL exists, increment phishing_rate
        new_rate = row[0] + 1
        c.execute('UPDATE url_data SET phishing_rate = ? WHERE url = ?', (new_rate, url))
    else:
        # URL does not exist, insert new record
        c.execute('INSERT INTO url_data (url, phishing_rate) VALUES (?, ?)', (url, 1))
    
    conn.commit()
    conn.close()

def get_all_email_data():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM email_data')
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_url_data():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('SELECT * FROM url_data')
    rows = c.fetchall()
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()
