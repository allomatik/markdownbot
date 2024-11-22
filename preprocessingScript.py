import os
import re
from pathlib import Path
import markdown
from nltk.tokenize import sent_tokenize

# configuration
input_folder = r"C:\Users\campb\Jarvis\rawData"
output_file = "preprocessed_data.txt"
chunk_size = 300  # target chunk size in words
special_tokens = {"start": "<|startoftext|>", "end": "<|endoftext|>"}

# helper functions
def clean_markdown(content):
    """Converts markdown to plain text and removes unwanted formatting."""
    # convert markdown to HTML, then strip HTML tags
    html_content = markdown.markdown(content)
    clean_content = re.sub(r"<[^>]+>", "", html_content)
    
    # remove inline links, images, and extra whitespace
    clean_content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_content)  # [text](url) -> text
    clean_content = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", clean_content)  # ![alt](image_url) -> ""
    clean_content = re.sub(r"\s+", " ", clean_content)  # normalize spaces
    
    return clean_content.strip()

def split_into_chunks(text, chunk_size):
    """Splits text into chunks of roughly chunk_size words."""
    sentences = sent_tokenize(text)  # split into sentences
    chunks, current_chunk = [], []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > chunk_size:
            # finish current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk, word_count = [], 0
        current_chunk.append(sentence)
        word_count += len(words)

    # add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# main preprocessing function
def preprocess_notes(input_folder, output_file, chunk_size):
    """Reads markdown notes, processes them, and writes structured chunks to a file."""
    input_path = Path(input_folder)
    all_chunks = []

    for file_path in input_path.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as file:
            raw_content = file.read()
        
        # clean markdown content
        plain_text = clean_markdown(raw_content)

        # split into chunks
        chunks = split_into_chunks(plain_text, chunk_size)
        
        # add metadata and special tokens
        for chunk in chunks:
            formatted_chunk = (
                f"{special_tokens['start']}\n"
                f"Title: {file_path.stem}\n\n"
                f"{chunk}\n"
                f"{special_tokens['end']}"
            )
            all_chunks.append(formatted_chunk)

    # write to output file
    with open(output_file, "w", encoding="utf-8") as out_file:
        out_file.write("\n\n".join(all_chunks))
    
    print(f"Preprocessing complete. Output saved to {output_file}.")

# run preprocessing
preprocess_notes(input_folder, output_file, chunk_size)
