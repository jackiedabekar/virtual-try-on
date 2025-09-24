import pandas as pd
import json
import requests
import os
from urllib.parse import urlparse

# Configuration
EXCEL_FILE = "./visiting_cards/cards_link.xlsx"  # Path to your Excel file
COLUMN_NAME = "LINKS"  # Column containing the JSON with S3 URLs
OUTPUT_DIR = (
    "./visiting_cards/downloaded_cards"  # Root folder to store downloaded files
)


def create_output_directory(directory):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_file(url, output_path):
    """Download file from URL and save to output_path."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {output_path}")
        else:
            print(f"Failed to download {url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")


def get_file_extension(url):
    """Extract file extension from URL."""
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    return os.path.splitext(filename)[1]  # Returns '.jpg', '.png', etc.


def main():
    # Create output directory
    create_output_directory(OUTPUT_DIR)

    # Read Excel file
    try:
        df = pd.read_excel(EXCEL_FILE)
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return

    # Check if the specified column exists
    if COLUMN_NAME not in df.columns:
        print(f"Column '{COLUMN_NAME}' not found in Excel file.")
        return

    # Iterate through rows
    for index, row in df.iterrows():
        try:
            # Parse JSON data
            json_data = json.loads(row[COLUMN_NAME])
            if isinstance(json_data, list) and json_data:
                for item in json_data:
                    # Check for 'front' and 'back' keys
                    for key in ["front", "back"]:
                        url = item.get(key)
                        if url:
                            # Get file extension from URL
                            extension = get_file_extension(url)
                            # Create filename like card_1.jpg or card_1.png
                            filename = f"card_{index + 1}{extension}"
                            output_path = os.path.join(OUTPUT_DIR, filename)

                            # Download the file
                            download_file(url, output_path)
                        else:
                            print(f"No '{key}' URL found in row {index + 2}")
            else:
                print(f"Invalid JSON format in row {index + 2}")
        except json.JSONDecodeError:
            print(f"JSON parsing error in row {index + 2}: {row[COLUMN_NAME]}")
        except Exception as e:
            print(f"Error processing row {index + 2}: {str(e)}")


if __name__ == "__main__":
    main()
