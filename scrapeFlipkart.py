import json
import time
import os
from datetime import datetime

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Define product links for Flipkart
links = {
    "Apple iPhone 16 (Black, 128 GB)": "https://www.flipkart.com/apple-iphone-16-black-128-gb/p/itmb07d67f995271?pid=MOBH4DQFG8NKFRDY",
    "Samsung Galaxy Z Flip 6 5G (256GB, Mint)": "https://www.flipkart.com/samsung-galaxy-z-flip6-5g-mint-256-gb/p/itmef6672178d8d7?pid=MOBH2HG9E4NG2BXN",
    "Samsung Galaxy Z Fold 6 Smartphone": "https://www.flipkart.com/samsung-galaxy-z-fold6-5g-pink-256-gb/p/itmc5b0d65ae951b?pid=MOBH2HG9BAR2XV3Z",
    "Apple iPhone 16 Pro Max(256 GB)": "https://www.flipkart.com/apple-iphone-16-pro-max-desert-titanium-256-gb/p/itmb102b8a51100e?pid=MOBH4DQFTQHZAKAF",
    "OnePlus 13 ": "https://www.flipkart.com/oneplus-13-arctic-dawn-256-gb/p/itm7e57559a9aa18?pid=MOBH8CHPY6Y8PYEQ",
    "Apple Watch Series 9": "https://www.flipkart.com/apple-watch-series-9-gps/p/itmc9050bff918fd?pid=SMWGTC2YHN84CW4Y",
    "Apple MacBook Air Laptop": "https://www.flipkart.com/apple-2022-macbook-air-m2-8-gb-256-gb-ssd-mac-os-monterey-mly33hn-a/p/itm48f8f11263927?pid=COMGFB2GMCRXZG85",
    "Apple AirPods Pro (2nd Generation)": "https://www.flipkart.com/apple-airpods-pro-2nd-generation-magsafe-case-usb-c-bluetooth-headset/p/itm60c8f5a308352?pid=ACCGTCYRNADXCXNG",
    "NIKE Mens Jordan Stay Loyal 3 Running Shoes": "https://www.flipkart.com/nike-jordan-stay-loyal-3-sneakers-men/p/itm6d20571705871?pid=SHOGZNH7YEQCCN6Z",
    "Premium Aquatic Eau De Cologne": "https://www.flipkart.com/premium-eau-de-cologne-100-ml/p/itmf3wgvsjxz3eyh?pid=PEREPAUFEYCDBZ8U",
    "Samsung S24 Ultra 5G (Titanium Gray, 256GB)": "https://www.flipkart.com/samsung-galaxy-s24-ultra-5g-titanium-gray-256-gb/p/itm12ef5ea0212ed?pid=MOBGX2F3RQKKKTAW",
    "Redmi Note 13 Pro": "https://www.flipkart.com/redmi-note-13-pro-5g-coral-purple-128-gb/p/itm810ee84cdaac6?pid=MOBGWFHFGWZVVYSK",
    "L'Oreal Paris Hyaluron Moisture 72HR Moisture Filling Shampoo": "https://www.flipkart.com/l-oral-paris-hyaluron-moisture-72h-filling-shampoo-1l/p/itme39c53c557be9?pid=SMPGGDWPGV937WGZ",
    "Lakmé 9 to 5 Kajal Twin Pack": "https://www.flipkart.com/lakm-eyeconic-kajal-twin-pack/p/itmf9edrxqnh8rhs?pid=KJLF7VNFP2KXUCVQ",
    "Puma Women's Pacific Maze Sneaker": "https://www.flipkart.com/puma-pacific-maze-wn-s-running-shoes-women/p/itm54d515a0306aa?pid=SHOGFX6FJYYFG7E3",
    "GoPro Hero13 Special Bundle Sports and Action Camera": "https://www.flipkart.com/gopro-hero13-special-bundle-sports-action-camera/p/itme10a001f83f92?pid=SAYH4B88GZCCZUQX",
    "SAMSUNG Galaxy A34 5G (Awesome Silver, 128 GB)": "https://www.flipkart.com/samsung-galaxy-a34-5g-awesome-silver-128-gb/p/itm2dd3e0ff525a8",
    "Apple iPhone 14 Plus (Blue, 128 GB)": "https://www.flipkart.com/apple-iphone-14-plus-blue-128-gb/p/itmac8385391b02b",
    "Redmi Note 12 Pro": "https://www.flipkart.com/redmi-note-12-pro-5g-onyx-black-128-gb/p/itmbc9fd7adaa32a"
}

# Initialize CSV file if it doesn't exist
if not os.path.exists("flipkart_current_data.csv"):
    pd.DataFrame(columns=["product_name", "Price", "Discount", "Date", "source"]).to_csv("flipkart_current_data.csv", index=False)

def scrape_flipkart_product_data(link):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument("--log-level=3")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(link)
    product_data = {}
    time.sleep(5)

    # Extract product price
    try:
        price_elem = driver.find_element(By.CSS_SELECTOR, "div.Nx9bqj.CxhGGd")
        product_data["selling price"] = int("".join(price_elem.text.strip().replace("₹", "").replace(",", "")))
    except:
        product_data["selling price"] = 0

    # Extract discount
    try:
        discount_elem = driver.find_element(By.CSS_SELECTOR, "div.UkUFwK.WW8yVX>span")
        product_data["Discount"] = discount_elem.text.strip()
    except:
        product_data["Discount"] = "0%"

    # Add date
    product_data["Date"] = time.strftime("%Y-%m-%d")
    driver.quit()
    return product_data

# Main loop to scrape data for each product
for product_name, link in links.items():
    product_data = scrape_flipkart_product_data(link)

    # Load existing data from CSV file
    if os.path.exists("flipkart_current_data.csv"):
        price = json.loads(pd.read_csv("flipkart_current_data.csv").to_json(orient="records"))
    else:
        price = []

    # Append new data
    price.append(
        {
            "product_name": product_name,
            "Price": product_data["selling price"],
            "Discount": product_data["Discount"],
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "source": "Flipkart",
        }
    )

    # Save updated data to CSV file
    pd.DataFrame(price).to_csv("flipkart_current_data.csv", mode="w", header=True, index=False)

print("Scraping completed and data saved to CSV file.")