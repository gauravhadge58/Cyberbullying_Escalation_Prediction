import pandas as pd
import requests
import json
import uuid
import time
from datetime import datetime, timedelta

def populate():
    print("Loading formatted_train.csv...")
    try:
        df = pd.read_csv("g:/TY SEM II/ML_PROJECT/Datasets/formatted_train.csv")
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    # Select a diverse subset of 200 messages
    # Let's ensure there is a mix of normal and bullying
    df_bully = df[df["label"] == 1].head(50)
    df_normal = df[df["label"] == 0].head(150)
    
    sample_df = pd.concat([df_bully, df_normal]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We need to send them as a list of dictionaries to /api/predict
    messages_payload = []
    
    start_time = datetime.now() - timedelta(minutes=200)
    
    for i, row in sample_df.iterrows():
        # Keep consistent conversation IDs from the dataset so escalation is calculated
        messages_payload.append({
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "user_id": row["user_id"],
            "message": row["message"], # The text
            "timestamp": (start_time + timedelta(minutes=i)).isoformat() + "Z"
        })

    # Group into batches of 20 to avoid overwhelming the network
    batch_size = 20
    url = "http://localhost:5000/api/predict"
    
    print(f"Sending {len(messages_payload)} messages to {url} in batches of {batch_size}...")
    
    for i in range(0, len(messages_payload), batch_size):
        batch = messages_payload[i:i+batch_size]
        payload = {"messages": batch}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print(f"Batch {i//batch_size + 1} sent successfully. ({len(batch)} messages)")
            else:
                print(f"Error on batch {i//batch_size + 1}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Connection error: {e}")
            break
            
        # Small delay between batches
        time.sleep(1)
        
    print("Population complete!")

if __name__ == "__main__":
    populate()
