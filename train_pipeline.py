import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

def prepare_and_train():
    print("Loading final_hateXplain.csv...")
    try:
        df = pd.read_csv("g:/TY SEM II/ML_PROJECT/Datasets/final_hateXplain.csv")
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Map comment to message
    df = df.rename(columns={"comment": "message"})
    
    # Map label to 1 (bullying) or 0 (normal)
    # HateXplain labels: 'normal', 'offensive', 'hatespeech'
    df["label"] = df["label"].apply(lambda x: 1 if str(x).lower() in ["offensive", "hatespeech"] else 0)

    # We need: id, conversation_id, user_id, timestamp
    num_rows = len(df)
    print(f"Loaded {num_rows} rows. Generating conversation metadata...")
    
    # Generate synthetic IDs
    df["id"] = [f"msg_{i}" for i in range(num_rows)]
    
    # Group every 5 messages into one conversation
    messages_per_conv = 5
    df["conversation_id"] = [f"conv_{i // messages_per_conv}" for i in range(num_rows)]
    
    # Random users
    np.random.seed(42)
    users = [f"user_{i}" for i in range(1, 101)] # 100 random users
    df["user_id"] = np.random.choice(users, num_rows)
    
    # Timestamps (spaced by 1 minute)
    start_time = datetime.now() - timedelta(days=30)
    df["timestamp"] = [start_time + timedelta(minutes=i) for i in range(num_rows)]

    # Select columns
    final_df = df[["id", "conversation_id", "user_id", "timestamp", "message", "label"]]
    
    # ---------------------------------------------------------------------------
    # Data Augmentation: Add explicit "normal" messages to debias the model
    # ---------------------------------------------------------------------------
    neutral_messages = [
        "Hello!", "How are you doing?", "Good morning everyone.",
        "I hope you have a great day.", "The weather is nice today.",
        "Can you help me with this task?", "I am learning machine learning.",
        "What is your favorite book?", "I like the new design.",
        "Let's meet tomorrow at 5 PM.", "Thank you for the information.",
        "Could you please explain this?", "I agree with your point.",
        "That's an interesting perspective.", "Have a nice weekend!",
        "Muslim culture is very rich.", "Jewish traditions are ancient.",
        "African art is beautiful.", "Diversity makes us stronger.",
        "Everyone deserves respect.", "I love my community.",
        "Welcome to our group!", "Peace be upon you.",
        "Happy holidays to everyone celebrating.", "Let's work together.",
    ]
    
    # Repeat neutral messages to have a significant impact
    neutral_df = pd.DataFrame({
        "message": neutral_messages * 10,
        "label": 0,
        "user_id": "system_bot",
        "conversation_id": "conv_neutral",
        "timestamp": datetime.now().isoformat()
    })
    neutral_df["id"] = [f"msg_neutral_{i}" for i in range(len(neutral_df))]
    
    final_df = pd.concat([final_df, neutral_df], ignore_index=True)
    print(f"Added {len(neutral_df)} neutral messages for debiasing.")
    
    # Take a sub-sample if it's too large to train quickly, or just train on all (20k rows)
    # 20k rows is fine for LogisticRegression, but might take a few seconds. We'll train on all.
    out_path = "g:/TY SEM II/ML_PROJECT/Datasets/formatted_train.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Saved formatted dataset to {out_path}.")
    
    # Send to the API
    url = "http://localhost:5000/api/train"
    print(f"Sending dataset to {url} for training...")
    
    with open(out_path, "rb") as f:
        files = {"file": ("formatted_train.csv", f, "text/csv")}
        response = requests.post(url, files=files)
        
    if response.status_code == 200:
        print("Training successful!")
        print(response.json())
    else:
        print(f"Failed to train. Status: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    prepare_and_train()
