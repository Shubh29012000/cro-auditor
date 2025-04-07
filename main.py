from requests_html import HTMLSession
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Optional: Use GPT if you have an API key (you can skip or keep empty for now)
openai.api_key = "YOUR_API_KEY"

# Define expected landing page structure
elements_required = [
    {"name": "announcement_bar", "keywords": ["offer", "free shipping", "new drop", "sale"]},
    {"name": "header_menu", "tag": "header"},
    {"name": "product_image_carousel", "min_images": 5},
    {"name": "product_title", "tag": "h1"},
    {"name": "product_price", "regex": r'\₹?\s?\d+[\.,]?\d*'},
    {"name": "compare_price", "tag": ["del", "strike"]},
    {"name": "discount_percentage", "regex": r'\d+%\s*off'},
    {"name": "review_star", "class": ["star", "review", "rating"]},
    {"name": "add_to_cart_button", "text": ["add to cart"]},
    {"name": "buy_now_button", "text": ["buy now"]},
    {"name": "offer_section", "keywords": ["limited time", "extra", "combo"]},
    {"name": "delivery_time", "keywords": ["delivery", "dispatch", "estimated"]},
    {"name": "product_badges", "keywords": ["guarantee", "authentic", "safe"]},
    {"name": "product_description", "tag": "p"},
    {"name": "frequently_bought_together", "keywords": ["frequently bought", "bundle"]},
    {"name": "a_plus_content", "keywords": ["why us", "brand promise", "crafted"]},
    {"name": "reviews_with_images", "img_check": True, "min_reviews": 50}
]

def fetch_rendered_html(url):
    session = HTMLSession()
    r = session.get(url)
    r.html.render(timeout=20)
    return r.html.html

def fetch_and_analyze(url):
    try:
        html = fetch_rendered_html(url)
    except Exception as e:
        return [{"section": "Error", "status": "Failed", "details": str(e)}]

    soup = BeautifulSoup(html, 'html.parser')
    report = []
    page_text = soup.get_text()

    for element in elements_required:
        result = {"section": element["name"], "status": "Missing", "details": "Not found"}

        if "tag" in element:
            tags = element["tag"] if isinstance(element["tag"], list) else [element["tag"]]
            for tag in tags:
                if soup.find(tag):
                    result["status"] = "Found"
                    result["details"] = f"Tag: {tag} found"
                    break

        elif "regex" in element:
            if re.search(element["regex"], soup.text):
                result["status"] = "Found"
                result["details"] = "Pattern matched"

        elif "text" in element:
            buttons = soup.find_all(["button", "a"])
            for b in buttons:
                if any(keyword in b.get_text(strip=True).lower() for keyword in element["text"]):
                    result["status"] = "Found"
                    result["details"] = f"Button with text: {b.get_text(strip=True)[:30]}"
                    break

        elif "keywords" in element:
            texts = soup.find_all(text=True)
            for t in texts:
                if any(keyword.lower() in t.lower() for keyword in element["keywords"]):
                    result["status"] = "Found"
                    result["details"] = f"Keyword match: {t.strip()[:30]}"
                    break

        elif element["name"] == "product_image_carousel":
            images = soup.find_all("img")
            if len(images) >= element["min_images"]:
                result["status"] = "Found"
                result["details"] = f"{len(images)} images found"

        elif element["name"] == "reviews_with_images":
            images = soup.find_all("img")
            if len(images) >= element["min_reviews"]:
                result["status"] = "Found"
                result["details"] = f"{len(images)} review images found"

        elif "class" in element:
            for class_keyword in element["class"]:
                if soup.find(class_=re.compile(class_keyword)):
                    result["status"] = "Found"
                    result["details"] = f"Class match: {class_keyword}"
                    break

        report.append(result)

    suggestion = analyze_page_with_ml(page_text)
    report.append({"section": "ML_Suggestions", "status": "Insights", "details": suggestion})

    return report

def analyze_page_with_ml(page_text):
    suggestions = []
    ideal_sections = [
        "Highlight limited-time offers",
        "Include trust badges (authentic, money-back, etc.)",
        "Ensure mobile-optimized image carousels",
        "Display social proof and high-quality reviews",
        "Emphasize benefits, not just features"
    ]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([page_text] + ideal_sections)
    sims = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    for i, sim in enumerate(sims):
        if sim < 0.2:
            suggestions.append(f"Missing/Weak: {ideal_sections[i]}")
    return "; ".join(suggestions)

# Streamlit UI
def run_app():
    st.set_page_config(page_title="CRO Auditor", layout="wide")
    st.title("🧠 D2C Product Page CRO Auditor")
    url = st.text_input("🔗 Enter your product page URL:")
    if st.button("🚀 Analyze"):
        if url:
            with st.spinner("Analyzing... please wait"):
                result = fetch_and_analyze(url)
                for r in result:
                    st.markdown(f"**{r['section']}**: `{r['status']}` — {r['details']}")
        else:
            st.warning("Please enter a valid URL.")

if __name__ == "__main__":
    run_app()
