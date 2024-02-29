import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
import re
from transformers import pipeline

# Create a summarization pipeline globally
pipe = pipeline("summarization", model="cnicu/t5-small-booksum")

# Create a Flask application
app = Flask(__name__)

# Define a function to scrape news articles
def scrape_news(news_url, class_name):
    try:
        response = requests.get(news_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find the specific div with the provided class name
            div_element = soup.find('div', class_=class_name)
            if div_element:
                # Extract text content from the div and remove newline characters
                news_text = div_element.get_text(separator=' ').replace('\n', ' ')
                # Extract image URL from the img tag inside the div
                img_element = div_element.find('img')
                if img_element:
                    image_url = img_element['src']
                else:
                    image_url = None
                return {"text": news_text.strip(), "image_url": image_url}
            else:
                return {"error": f"Div element not found with the specified class name '{class_name}'"}
        else:
            return {"error": f"Unable to retrieve the webpage. Status code: {response.status_code}"}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

# Define API routes
@app.route('/api/scrape/news', methods=['POST'])  # Allow POST requests
def api_scrape_news():
    data = request.json
    news_url = data.get('news_url')
    class_name = data.get('class_name')
    
    if not news_url or not class_name:
        return jsonify({"error": "Please provide 'news_url' and 'class_name' in the request payload"})
    
    scraped_data = scrape_news(news_url, class_name)

    if 'error' in scraped_data:
        return jsonify(scraped_data)

    # Use the global pipeline for summarization
    summary = pipe(scraped_data["text"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, no_repeat_ngram_size=2)
    summary_text = summary[0]['summary_text']

    # Extract date and clean text
    date_pattern = r'\(([^)]+)\)$'
    match = re.search(date_pattern, scraped_data["text"])
    if match:
        date_str = match.group(1)
        cleaned_text = re.sub(date_pattern, '', scraped_data["text"]).replace('Mwebantu', '').replace('\n', ' ').strip()
    else:
        date_str = None
        cleaned_text = scraped_data["text"].replace('Mwebantu', '').replace('\n', ' ').strip()

    # Create cleaned data dictionary
    cleaned_data = {
        "text": cleaned_text,
        "date": date_str,
        "image_url": scraped_data["image_url"],
        "summary_text": summary_text
    }

    return jsonify(cleaned_data)

if __name__ == '__main__':
    app.run(debug=True)
