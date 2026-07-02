import re

import numpy as np

def extract_email_body(msg):
    """
    Extract text body from email message object.
    
    Handles both simple and multipart messages.
    For multipart messages, extracts text/plain parts and converts HTML to text.
    
    Args:
        msg: email.message.Message object
    
    Returns:
        Extracted text string
    """
    body_text = ""
    
    # Check if message is multipart
    if msg.is_multipart():
        # Iterate through all parts
        for part in msg.walk():
            content_type = part.get_content_type()
            
            # Get text/plain parts
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_text += payload.decode('utf-8', errors='ignore') + " "
                except:
                    pass
            
            # Get text/html parts and strip basic HTML
            elif content_type == "text/html":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_text = payload.decode('utf-8', errors='ignore')
                        # Basic HTML tag removal
                        html_text = re.sub(r'<[^>]+>', ' ', html_text)
                        body_text += html_text + " "
                except:
                    pass
    else:
        # Single part message
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                body_text = payload.decode('utf-8', errors='ignore')
            else:
                # If decode fails, try getting as string
                body_text = str(msg.get_payload())
        except:
            body_text = str(msg.get_payload())
    
    return body_text

def preprocess_email(msg, include_subject=True):
    """
    Clean and preprocess a single email.
    
    Preprocessing steps:
    1. Extract email body (and optionally subject)
    2. Remove HTML tags
    3. Convert to lowercase
    4. Remove URLs
    5. Remove email addresses
    6. Remove numbers (optional - currently keeping them)
    7. Remove special characters and punctuation
    8. Remove extra whitespace
    9. Basic stopword removal
    
    Args:
        msg: email.message.Message object
        include_subject: Whether to include subject line in text
    
    Returns:
        Cleaned text string
    """
    # Extract body text
    text = extract_email_body(msg)
    
    # Optionally add subject line (subjects often contain spam indicators)
    if include_subject:
        subject = msg.get('Subject', '')
        if subject:
            text = subject + " " + text
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (http://, https://, www.)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove numbers (you can keep them if you think they're useful for spam detection)
    # Uncomment the next line to remove numbers
    # text = re.sub(r'\d+', ' ', text)
    
    # Remove special characters and punctuation, keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Basic stopword removal (common English words that don't add much meaning)
    # Using a simple list to avoid NLTK dependency
    # You can expand this list or use NLTK's stopwords for more comprehensive removal
    stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
        'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
        'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once'
    }
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Simple stemming (remove common suffixes)
    # For better stemming, use NLTK's PorterStemmer or LancasterStemmer
    def simple_stem(word):
        """Apply very basic stemming rules."""
        # Remove common suffixes
        if word.endswith('ing'):
            word = word[:-3]
        elif word.endswith('ed'):
            word = word[:-2]
        elif word.endswith('ly'):
            word = word[:-2]
        elif word.endswith('es'):
            word = word[:-2]
        elif word.endswith('s') and len(word) > 3:
            word = word[:-1]
        return word
    
    # Apply simple stemming to each word
    words = [simple_stem(word) for word in words]
    
    # Join words back into string
    text = ' '.join(words)
    
    return text

def preprocess_all_emails(emails, include_subject=True):
    """
    Apply preprocessing to all emails.
    
    Args:
        emails: list of email.message.Message objects
        include_subject: Whether to include subject lines
    
    Returns:
        cleaned_emails: list of cleaned text strings
        empty_indices: list of indices where emails became empty after cleaning
    """
    print(f"Preprocessing {len(emails)} emails...")
    
    cleaned_emails = []
    empty_indices = []
    
    for i, msg in enumerate(emails):
        try:
            cleaned_text = preprocess_email(msg, include_subject=include_subject)
            cleaned_emails.append(cleaned_text)
            
            # Track empty emails
            if len(cleaned_text.strip()) == 0:
                empty_indices.append(i)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(emails)} emails...")
                
        except Exception as e:
            print(f"  Error processing email {i}: {e}")
            # Add empty string for failed emails
            cleaned_emails.append("")
            empty_indices.append(i)
    
    print(f"Preprocessed {len(cleaned_emails)} emails ({len(empty_indices)} empty after cleaning)")
    
    if empty_indices:
        print(f"  Empty indices (first 10): {empty_indices[:10]}")
    
    return cleaned_emails, empty_indices

def display_preprocessing_examples(emails, labels, cleaned_emails, num_examples=2):
    """
    Display before/after examples of preprocessing.
    
    Args:
        emails: original email objects
        labels: email labels
        cleaned_emails: preprocessed email texts
        num_examples: number of examples to show per class
    """
    print("\n" + "="*80)
    print("PREPROCESSING EXAMPLES")
    print("="*80)
    
    # Show ham examples
    ham_indices = np.where(labels == 0)[0]
    print("\n--- HAM EMAIL EXAMPLES ---")
    for i in range(min(num_examples, len(ham_indices))):
        idx = ham_indices[i]
        print(f"\nExample {i+1}:")
        print(f"Subject: {emails[idx].get('Subject', 'No Subject')}")
        
        original_body = extract_email_body(emails[idx])
        print(f"\nOriginal (first 300 chars):")
        print(original_body[:300] + "...")
        
        print(f"\nCleaned (first 300 chars):")
        print(cleaned_emails[idx][:300] + "...")
        print("-" * 80)
    
    # Show spam examples
    spam_indices = np.where(labels == 1)[0]
    print("\n--- SPAM EMAIL EXAMPLES ---")
    for i in range(min(num_examples, len(spam_indices))):
        idx = spam_indices[i]
        print(f"\nExample {i+1}:")
        print(f"Subject: {emails[idx].get('Subject', 'No Subject')}")
        
        original_body = extract_email_body(emails[idx])
        print(f"\nOriginal (first 300 chars):")
        print(original_body[:300] + "...")
        
        print(f"\nCleaned (first 300 chars):")
        print(cleaned_emails[idx][:300] + "...")
        print("-" * 80)

