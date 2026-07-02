import email
from collections import Counter
from pathlib import Path

import numpy as np

from classifier.config import DATA_DIR

def load_emails(base_path=None):
    """
    Load all emails from the SpamAssassin dataset.
    
    Args:
        base_path: Path to SpamAssassin dataset directory (default: DATA_DIR)
    
    Returns:
        emails: list of email message objects
        labels: numpy array of labels (0=ham, 1=spam)
        raw_texts: list of raw email text strings
    """
    if base_path is None:
        base_path = DATA_DIR
    base_path = Path(base_path)
    emails = []
    labels = []
    raw_texts = []
    
    # Define directories and their labels
    # ham = 0, spam = 1
    email_categories = {
        'easy_ham': 0,
        'easy_ham_2': 0,
        'hard_ham': 0,
        'spam': 1,
        'spam_2': 1
    }
    
    print(f"Loading emails from {base_path}...")
    
    for category, label in email_categories.items():
        category_path = base_path / category
        
        if not category_path.is_dir():
            print(f"Warning: Directory {category_path} not found. Skipping...")
            continue
        
        # Get all files in the category directory
        files = [f for f in category_path.iterdir() if f.is_file() and not f.name.endswith('.ipynb')]
        
        print(f"  Loading {len(files)} emails from {category}...")
        
        for file_path in files:
            try:
                # Read email file with latin-1 encoding to handle special characters
                with open(file_path, 'r', encoding='latin-1') as file_handler:
                    # Parse email using email library
                    msg = email.message_from_file(file_handler)
                    
                    # Store the parsed message object
                    emails.append(msg)
                    labels.append(label)
                    
                    # Also store raw text for reference
                    file_handler.seek(0)
                    raw_texts.append(file_handler.read())
                    
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                continue
    
    print(f"\nTotal emails loaded: {len(emails)}")
    
    return emails, np.array(labels), raw_texts

def explore_dataset(emails, labels):
    """
    Perform initial exploration of the dataset.
    
    Args:
        emails: list of email message objects
        labels: numpy array of labels
    """
    print("\n" + "="*80)
    print("DATASET EXPLORATION")
    print("="*80)
    
    # Count total samples
    total_samples = len(emails)
    print(f"\nTotal samples: {total_samples}")
    
    # Calculate class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\nClass Distribution:")
    print(f"  Ham (0):  {class_dist.get(0, 0):5d} samples ({class_dist.get(0, 0)/total_samples*100:.2f}%)")
    print(f"  Spam (1): {class_dist.get(1, 0):5d} samples ({class_dist.get(1, 0)/total_samples*100:.2f}%)")
    
    # Check for empty emails
    empty_count = 0
    for msg in emails:
        payload = msg.get_payload()
        if isinstance(payload, str) and len(payload.strip()) == 0:
            empty_count += 1
    
    print(f"\nEmpty emails: {empty_count}")
    
    # Analyze content types
    content_types = Counter()
    for msg in emails:
        content_types[msg.get_content_type()] += 1
    
    print(f"\nContent Types:")
    for content_type, count in content_types.most_common(10):
        print(f"  {content_type:30s}: {count:5d} ({count/total_samples*100:.2f}%)")
    
    # Display sample emails
    print("\n" + "="*80)
    print("SAMPLE EMAILS")
    print("="*80)
    
    # Show one ham and one spam example
    ham_idx = np.where(labels == 0)[0][0]
    spam_idx = np.where(labels == 1)[0][0]
    
    print("\n--- SAMPLE HAM EMAIL ---")
    print(f"Subject: {emails[ham_idx].get('Subject', 'No Subject')}")
    print(f"From: {emails[ham_idx].get('From', 'Unknown')}")
    print(f"Content-Type: {emails[ham_idx].get_content_type()}")
    payload = emails[ham_idx].get_payload()
    if isinstance(payload, str):
        print(f"Body preview: {payload[:200]}...")
    else:
        print(f"Body: [Multipart message]")
    
    print("\n--- SAMPLE SPAM EMAIL ---")
    print(f"Subject: {emails[spam_idx].get('Subject', 'No Subject')}")
    print(f"From: {emails[spam_idx].get('From', 'Unknown')}")
    print(f"Content-Type: {emails[spam_idx].get_content_type()}")
    payload = emails[spam_idx].get_payload()
    if isinstance(payload, str):
        print(f"Body preview: {payload[:200]}...")
    else:
        print(f"Body: [Multipart message]")
    
    print("\n" + "="*80)

def remove_empty_emails(emails, cleaned_emails, labels, raw_texts, empty_indices):
    """
    Remove empty emails from the dataset.
    
    After preprocessing, some emails may be empty. This function removes them
    to ensure the BOW vectorizer doesn't encounter issues.
    
    Args:
        emails: list of email message objects
        cleaned_emails: list of cleaned email texts
        labels: numpy array of labels
        raw_texts: list of raw email texts
        empty_indices: list of indices of empty emails
    
    Returns:
        Filtered versions of all inputs
    """
    if len(empty_indices) == 0:
        print("\nNo empty emails to remove.")
        return emails, cleaned_emails, labels, raw_texts
    
    print(f"\nRemoving {len(empty_indices)} empty email(s) from dataset...")
    print(f"Empty email indices: {empty_indices}")
    
    # Create boolean mask for non-empty emails
    mask = np.ones(len(emails), dtype=bool)
    mask[empty_indices] = False
    
    # Filter all arrays
    filtered_emails = [email for i, email in enumerate(emails) if mask[i]]
    filtered_cleaned = [text for i, text in enumerate(cleaned_emails) if mask[i]]
    filtered_labels = labels[mask]
    filtered_raw = [text for i, text in enumerate(raw_texts) if mask[i]]
    
    print(f"Dataset size: {len(emails)} -> {len(filtered_emails)}")
    print(f"Removed: {len(emails) - len(filtered_emails)} emails")
    
    return filtered_emails, filtered_cleaned, filtered_labels, filtered_raw

