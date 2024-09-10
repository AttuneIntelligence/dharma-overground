import sys, os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def structure_dharma(dharma_df,
                     output_file="./dho_message_threads.json",
                     force_rebuild=False):

    ### CHECK IF ALREADY EXISTS...
    if os.path.exists(output_file) and not force_rebuild:
        print(f"{output_file} already exists. Reading data.")
        with open(output_file) as json_data:
            text_threads = json.loads(json_data.read())
        return text_threads

    ### SORT POSTS BY MESSAGE THREAD AND BY TIMESTAMP
    df_sorted = dharma_df.sort_values(['thread_id', 'timestamp'])
    thread_dict = {}

    print('Organizing Dharma-Overground messages into appropriate message threads...')
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted)):
        thread_id = row['thread_id']
        message = {
            'user_id': row['user_id'],
            'timestamp': row['timestamp'],
            'title': row['title'],
            'post': row['post']
        }
        if thread_id not in thread_dict:
            thread_dict[thread_id] = []
        thread_dict[thread_id].append(message)

    ### THEN SORT THOSE INTO TEXT STRINGS
    text_threads = {}
    print('Saving messages as text threads...')
    for thread_id, messages in tqdm(thread_dict.items(), total=len(thread_dict.items())):
        title = messages[0]['title']
        thread_text = f"# {title}\n\n"
        thread_users = set()
        for message in messages:
            thread_text += f"    User {message['user_id']}: {message['post']}\n\n"
            thread_users.add(message['user_id'])
        text_threads[thread_id] = {
            'text': thread_text,
            'users': list(thread_users),
            'token_len': Agent.OAI.get_token_len(thread_text),
            'title': title,
            'date_of_first_message': messages[0]['timestamp'],
            'date_of_last_message': messages[-1]['timestamp'],
            'thread_id': thread_id
        }

    ### SAVE TO FILE OF TEXT THREADS
    print('Writing to intermediate JSON file...')
    with open(output_file, 'w') as f:
        json.dump(text_threads, f)
    print('Done!')
    
    return text_threads

def visualize_token_length_distribution(text_threads):
    # Extract token lengths from the text_threads dictionary
    token_lengths = [thread_info['token_len'] for thread_info in text_threads.values()]

    # Calculate statistics
    mean_length = np.mean(token_lengths)
    median_length = np.median(token_lengths)
    max_length = max(token_lengths)
    min_length = min(token_lengths)
    q1, q3 = np.percentile(token_lengths, [25, 75])

    # Create the histogram
    plt.figure(figsize=(14, 8))
    
    # Use log-spaced bins
    bins = np.logspace(np.log10(max(1, min_length)), np.log10(max_length), num=100)
    
    n, bins, patches = plt.hist(token_lengths, bins=bins, edgecolor='black')

    # Set x-axis to logarithmic scale
    plt.xscale('log')

    # Add labels and title
    plt.xlabel('Token Length (log scale)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Token Lengths in Dharma-Overground Threads')

    # Add statistics to the plot
    plt.axvline(mean_length, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean_length:.2f}')
    plt.axvline(median_length, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median_length:.2f}')
    plt.axvline(q1, color='orange', linestyle='dashed', linewidth=2, label=f'Q1: {q1:.2f}')
    plt.axvline(q3, color='purple', linestyle='dashed', linewidth=2, label=f'Q3: {q3:.2f}')
    plt.axvline(max_length, color='b', linestyle='dashed', linewidth=2, label=f'Max: {max_length}')

    # Add a legend
    plt.legend()

    # Add text with additional statistics
    stats_text = f'Total Threads: {len(token_lengths)}\n'

    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust x-axis ticks for better readability
    plt.xticks([10, 100, 1000, 10000, 100000], ['10', '100', '1K', '10K', '100K'])

    # Show the plot
    plt.tight_layout()
    plt.show()

def filter_threads_by_token_length(text_threads, 
                                   min_tokens=1000,
                                   max_tokens=9000):
    """
    FILTER DHARMA OVERGROUND THREADS BY THEIR TOKEN LENGTH
    """
    filtered_threads = {}
    
    for thread_id, thread_info in text_threads.items():
        token_len = thread_info['token_len']
        if min_tokens <= token_len <= max_tokens:
            filtered_threads[thread_id] = thread_info
    
    print(f"Filtered {len(filtered_threads)} threads out of {len(text_threads)} total message threads.")
    print(f"{round((len(filtered_threads)/len(text_threads))*100, 1)}% of original threads remain in filtered dataset.")
    print(f"Token length range: {min_tokens} to {max_tokens}")
    
    return filtered_threads

