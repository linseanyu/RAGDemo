import pandas as pd
import random
from datetime import datetime, timedelta
import uuid

# Seed for reproducibility
random.seed(42)

# --- Generate products.csv ---
# Product categories and templates
categories = ["Electronics", "Clothing", "Home Goods"]
product_templates = {
    "Electronics": [
        ("Wireless Headphones", "Noise-canceling, {battery}-hour battery life", 50, 200),
        ("Smart Watch", "{feature} tracking, waterproof, {battery}-day battery", 100, 300),
        ("Bluetooth Speaker", "Portable, {feature} sound, {battery}-hour battery", 30, 150),
        ("Laptop", "{feature} processor, {storage}GB storage", 500, 2000),
        ("Smartphone", "{feature} camera, {storage}GB storage", 300, 1200)
    ],
    "Clothing": [
        ("T-Shirt", "{feature} cotton, available in {color}", 10, 50),
        ("Jacket", "Water-resistant, {feature} lining", 40, 150),
        ("Sneakers", "{feature} sole, available in {color}", 30, 120),
        ("Jeans", "{feature} fit, {color} denim", 20, 80),
        ("Sweater", "{feature} wool, {color} design", 25, 100)
    ],
    "Home Goods": [
        ("Coffee Maker", "{feature} brewing, {capacity}-cup capacity", 20, 100),
        ("Vacuum Cleaner", "{feature} suction, cordless", 50, 300),
        ("Air Fryer", "{feature} cooking, {capacity}-liter capacity", 40, 200),
        ("Bedding Set", "{feature} fabric, {color} pattern", 30, 150),
        ("Table Lamp", "{feature} design, {color} finish", 15, 80)
    ]
}

colors = ["black", "white", "blue", "red", "green"]
features = {
    "Electronics": ["high-performance", "advanced", "compact", "ultra-fast", "AI-powered"],
    "Clothing": ["soft", "breathable", "slim", "cozy", "durable"],
    "Home Goods": ["modern", "efficient", "stylish", "premium", "smart"]
}
batteries = [10, 12, 15, 20, 24, 7, 5]
storages = [64, 128, 256, 512]
capacities = [2, 4, 6, 8]

# Generate product data
products = []
for i in range(1000):
    category = random.choice(categories)
    template = random.choice(product_templates[category])
    name, desc_template, min_price, max_price = template
    feature = random.choice(features[category])
    color = random.choice(colors)
    battery = random.choice(batteries) if "{battery}" in desc_template else ""
    storage = random.choice(storages) if "{storage}" in desc_template else ""
    capacity = random.choice(capacities) if "{capacity}" in desc_template else ""
    description = desc_template.format(feature=feature, color=color, battery=battery, storage=storage, capacity=capacity)
    price = round(random.uniform(min_price, max_price), 2)
    products.append({
        "product_id": i + 1,
        "name": name,
        "description": description,
        "price": price
    })

products_df = pd.DataFrame(products)
products_df.to_csv("products.csv", index=False)

# --- Generate conversations.csv ---
# Conversation templates tied to products
conversation_templates = [
    ("What is the battery life of the {name}?", "The {name} has a {battery}-hour battery life."),
    ("Is the {name} waterproof?", "Yes, the {name} is waterproof and suitable for {activity}."),
    ("What colors is the {name} available in?", "The {name} is available in {color} and other colors."),
    ("Does the {name} have {feature} features?", "Yes, the {name} includes {feature} features for enhanced performance."),
    ("Is the {name} in stock?", "The {name} is currently in stock and available for ${price}.")
]
activities = ["swimming", "outdoor use", "daily wear"]

# Generate conversation data
conversations = []
start_date = datetime(2025, 1, 1)
for i in range(1000):
    product = random.choice(products)
    name = product["name"]
    price = product["price"]
    template = random.choice(conversation_templates)
    customer_msg, agent_response = template
    feature = random.choice(features[random.choice(categories)])
    color = random.choice(colors)
    battery = random.choice(batteries) if "battery" in agent_response else ""
    activity = random.choice(activities) if "{activity}" in agent_response else ""
    customer_msg = customer_msg.format(name=name, feature=feature)
    agent_response = agent_response.format(name=name, battery=battery, color=color, feature=feature, price=price, activity=activity)
    timestamp = start_date + timedelta(minutes=random.randint(0, 365 * 24 * 60))
    conversations.append({
        "conversation_id": str(uuid.uuid4()),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
        "customer_message": customer_msg,
        "agent_response": agent_response
    })

conversations_df = pd.DataFrame(conversations)
conversations_df.to_csv("conversations.csv", index=False)

print("Generated products.csv and conversations.csv with 1000 records each.")