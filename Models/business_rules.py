# save_business_rules.py
import json
import os

# set thresholds
business_rules = {
    "outlier_thresholds": {
        "monetary": 3799.39,   
        "frequency": 11       
    },
    "cluster_mapping": {
        "0": "Regular",
        "1": "Lapsed", 
        "2": "Occasional",
        "3": "Premium",
        "A": "High_Spender",
        "B": "Power_Shopper", 
        "C": "Elite"
    },
    "cluster_descriptions": {
        "Regular": "Steady customers with moderate buying patterns",
        "Lapsed": "Inactive customers needing re-engagement",
        "Occasional": "Infrequent buyers with low spending",
        "Premium": "Consistent buyers with moderate spending",
        "High_Spender": "Customers spending over $3,799 (big spenders)",
        "Power_Shopper": "Customers with over 11 purchases (very active)",
        "Elite": "VIP: Both high spending AND high frequency"
    }
}

# Save to JSON file
with open('Model/business_rules.json', 'w') as f:
    json.dump(business_rules, f, indent=4)
