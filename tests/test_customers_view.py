import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from analytics.drivers import compute_dimension_drivers


def test_customer_dimension_heuristic_detection():
    # Customer-shaped names
    customer_cols = ["customer", "client_name", "account_id", "user_segment", "buyer_type", "consumer_group"]
    pattern = re.compile(r"(customer|client|account|user|buyer|consumer|segment)", re.I)

    for col in customer_cols:
        assert bool(pattern.search(col)), f"{col} should match customer heuristic"

    # Non-customer names
    non_customer_cols = ["region", "warehouse", "product_category", "shipping_method", "vendor"]
    for col in non_customer_cols:
        assert not bool(pattern.search(col)), f"{col} should not match customer heuristic"
