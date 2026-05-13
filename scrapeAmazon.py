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

# Define product links
links = {
    "Samsung 80 cm (32 inches) HD Ready Smart LED TV UA32T4380AKXXL (Glossy Black)": "https://amzn.in/d/5Z9eft3",
    "Apple iPhone 16 (Black, 128 GB)": "https://amzn.in/d/7ibZClF",
    "Samsung Galaxy Z Flip 6 5G (256GB, Mint)": "https://amzn.in/d/fSnWU2X",
    "Samsung Galaxy Z Fold 6 Smartphone": "https://amzn.in/d/fIWZVoA",
    "Apple iPhone 16 Pro Max(256 GB)": "https://amzn.in/d/1nq4PwX",
    "OnePlus 13": "https://amzn.in/d/4lFIWoT",
    "Apple Watch Series 9": "https://amzn.in/d/2yJ2sfr",
    "Apple MacBook Air Laptop": "https://amzn.in/d/3CTI4pr",
    "Apple AirPods Pro (2nd Generation)": "https://amzn.in/d/7UiAcoh",
    "NIKE Mens Jordan Stay Loyal 3 Running Shoes": "https://amzn.in/d/j4hxieV",
    "Premium Aquatic Eau De Cologne": "https://amzn.in/d/cd8oT7T",
    "Samsung S24 Ultra 5G (Titanium Gray, 256GB)": "https://amzn.in/d/j5fjbz8",
    "Redmi Note 13 Pro": "https://amzn.in/d/60RSLUR",
    "L'Oreal Paris Hyaluron Moisture 72HR Moisture Filling Shampoo": "https://amzn.in/d/e38Y8L8",
    "Lakmé 9 to 5 Kajal Twin Pack": "https://amzn.in/d/cAz42m0",
    "Puma Women's Pacific Maze Sneaker": "https://amzn.in/d/f4QhIhg",
    "Noise Pro 6 Max Smart Watch": "https://amzn.in/d/93MA7yU",
    "Crompton Optimus 100 Litres Desert Air Cooler for home": "https://amzn.in/d/8T6mZIS",
    "Godrej 272 L 3 Star Convertible Technology, 30 days Farms Freshness Inverter Frost Free Double Door Refrigerator": "https://amzn.in/d/75lZXnl",
    "Nothing Phone (3A) 5G (Black, 8GB RAM, 256GB Storage)": "https://amzn.in/d/4EcK09L",
    "GoPro Hero13 Special Bundle Sports and Action Camera": "https://amzn.in/d/adcIsVS",
    "LG 322 L 3 Star Frost-Free Smart Inverter Double Door Refrigerator": "https://amzn.in/d/01AEIEL",
    "Sony WH-1000XM5 Wireless Noise Canceling Headphones": "https://amzn.in/d/7WA7yIX",
    "Redmi Note 12 Pro": "https://amzn.in/d/iTfccbP",
    "Apple iPhone 14 Plus (Blue, 128 GB)": "https://amzn.in/d/dKAwUMK",
    "SAMSUNG Galaxy A34 5G (Awesome Silver, 128 GB)": "https://amzn.in/d/3cKhmEM"
}

# Initialize CSV file if it doesn't exist
if not os.path.exists("amazon_current_data.csv"):
    pd.DataFrame(columns=["Product name", "Price", "Discount", "Date", "source"]).to_csv("amazon_current_data.csv", index=False)

def scrape_product_data(link):
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    )
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    driver = webdriver.Chrome(options=options)

    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })

    driver.get(link)
    time.sleep(4)

    product_data = {}
    # continue extraction...


    # Extract product price
    try:
        price_elem = driver.find_element(
            By.XPATH,
            '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]',
        )
        product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))
    except:
        try:
            price_elem = driver.find_element(By.CSS_SELECTOR, ".a-price-whole")
            product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))
        except:
            product_data["selling price"] = 0

    # Extract original price
    try:
        original_price = driver.find_element(By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]').text
        product_data["original price"] = int("".join(original_price.strip().split(",")))
    except:
        product_data["original price"] = 0

    # Extract discount
    try:
        discount = driver.find_element(By.XPATH, '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]')
        product_data["Discount"] = discount.text.strip()
    except:
        product_data["Discount"] = "0%"

    # Add date
    product_data["Date"] = time.strftime("%Y-%m-%d")
    driver.quit()
    return product_data

# Main loop to scrape data for each product
for product_name, link in links.items():
    product_data = scrape_product_data(link)

    # Load existing data from CSV file
    try:
        df = pd.read_csv("amazon_current_data.csv")
        price = df.to_dict(orient="records")
    except Exception as e:
        print("CSV load error:", e)
        price = []

    # Append new data
    price.append(
        {
            "product_name": product_name,
            "Price": product_data["selling price"],
            "Discount": product_data["Discount"],
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "source": "Amazon",
        }
    )

    # Save updated data to CSV file
    pd.DataFrame(price).to_csv("amazon_current_data.csv", mode="w", header=True, index=False)

print("Scraping completed and data saved to CSV file.")