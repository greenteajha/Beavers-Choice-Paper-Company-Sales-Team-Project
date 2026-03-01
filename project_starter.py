import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, Dict[str, int]]:
    """
    Retrieve a snapshot of available inventory as of a specific date,
    including the minimum stock threshold for each item.

    This function calculates the net quantity of each item by summing
    all stock orders and subtracting all sales up to and including the given date,
    then joins with the inventory table to include each item's min_stock_level.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary mapping item names to a dict containing
                                   'current_stock' and 'min_stock_level'.
    """
    # --- Previous query (current_stock only, no threshold) ---
    # query = """
    #     SELECT
    #         item_name,
    #         SUM(CASE
    #             WHEN transaction_type = 'stock_orders' THEN units
    #             WHEN transaction_type = 'sales' THEN -units
    #             ELSE 0
    #         END) as stock
    #     FROM transactions
    #     WHERE item_name IS NOT NULL
    #     AND transaction_date <= :as_of_date
    #     GROUP BY item_name
    #     HAVING stock > 0
    # """
    # result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})
    # return dict(zip(result["item_name"], result["stock"]))

    # SQL query to compute stock levels and join with inventory thresholds
    query = """
        SELECT
            t.item_name,
            SUM(CASE
                WHEN t.transaction_type = 'stock_orders' THEN t.units
                WHEN t.transaction_type = 'sales' THEN -t.units
                ELSE 0
            END) as current_stock,
            i.min_stock_level
        FROM transactions t
        LEFT JOIN inventory i ON t.item_name = i.item_name
        WHERE t.item_name IS NOT NULL
        AND t.transaction_date <= :as_of_date
        GROUP BY t.item_name
        HAVING current_stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a nested dictionary {item_name: {current_stock, min_stock_level}}
    return {
        row["item_name"]: {
            "current_stock": int(row["current_stock"]),
            "min_stock_level": int(row["min_stock_level"]) if pd.notna(row["min_stock_level"]) else 0
        }
        for _, row in result.iterrows()
    }

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv(dotenv_path=".env")
openai_api_key = os.getenv("UDACITY_API_KEY")

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_base="https://openai.vocareum.com/v1",
    api_key=openai_api_key,
)

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent
@tool
def get_all_inventory_tool(as_of_date: str) -> Dict[str, Dict[str, int]]:
    """
    Use this tool to get a high-level snapshot of ALL items currently in stock as of a given date,
    including each item's minimum stock threshold for reorder decisions.

    Call this tool when:
    - You need to monitor overall stock health and identify items that are below their reorder threshold.
    - You need a global inventory view to decide whether any items should be reordered.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, Dict[str, int]]: A dictionary mapping each item name to a dict with:
            - 'current_stock': the current quantity available.
            - 'min_stock_level': the minimum threshold below which a reorder should be triggered.
        Only items with current_stock > 0 are included. Items missing from the result have zero stock.
    """

    return get_all_inventory(as_of_date)


@tool
def get_stock_level_tool(item_name: str, as_of_date: Union[str, datetime]) -> int:
    """
    Use this tool to get the precise current stock count for a specific item by exact item name.

    Call this tool when:
    - You have fuzzy-matched a customer's item request to an exact item name in our catalog.
    - You need to confirm whether we have enough stock to fulfill the customer's requested quantity.
    - You need the stock count to pair with historical pricing when building a quote.

    Use search_quote_history_tool first to find the exact item name from past quotes, then
    call this tool with that matched name to get the current stock level.

    Args:
        item_name (str): The exact item name as it appears in the catalog.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        int: Current stock count for the item. Returns 0 if the item has no stock.
    """

    return get_stock_level(item_name, as_of_date)["current_stock"].iloc[0]


@tool
def get_supplier_delivery_date_tool(input_date_str: str, quantity: int) -> str:
    """
    Use this tool:
        - When your stock is low and need to consider reordering.
        - When you need to tell the customer when an item could be available.
    
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """

    return get_supplier_delivery_date(input_date_str, quantity)
    

# Tools for sales agent

@tool
def generate_financial_report_tool(as_of_date: Union[str, datetime]) -> Dict:
    """
    Use this tool when you need access to financial context like revenue, costs, profit, and inventory values when deciding how to price a quote.
    
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    return generate_financial_report(as_of_date)

# Tools for quoting agent

@tool
def search_quote_history_tool(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Use this tool when you need to reference historical quotes for a specific search term before generating a new quote.
    
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    return search_quote_history(search_terms, limit)

# Tools for sales agent and inventory agent

@tool
def get_cash_balance_tool(as_of_date: Union[str, datetime]) -> float:
    """
    Use this tool when you need the cash balance as of date.

    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    return get_cash_balance(as_of_date)

@tool
def create_transaction_tool(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    Use this tool when you would like to create a transaction.

    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    return create_transaction(item_name, transaction_type, quantity, price, date)


# Set up your agents and create an orchestration agent that will manage them.

class InventoryAgent(ToolCallingAgent):
    """Agent responsible for fuzzy-matching requests, checking stock, and making reorder decisions."""

    def __init__(self, model):
        super().__init__(
            tools=[get_all_inventory_tool, get_cash_balance_tool, get_supplier_delivery_date_tool],
            model=model,
            name="inventory_agent",
            description="Normalises customer item names against the live catalog (which already contains current_stock and min_stock_level per item), checks both thresholds directly from the catalog result, and runs a reorder feasibility check (cash balance first, then supplier delivery date vs customer deadline) to return a per-item fulfillment report."
        )

class QuotingAgent(ToolCallingAgent):
    """Agent responsible for pricing fulfillable items with stock-aware tiered discounts.

    In normal operation, generate_quote handles pricing entirely in Python (price lookup
    via search_quote_history + deterministic discount formula). This agent is only invoked
    as a fallback for items with no historical price data, where it estimates a price.
    """

    def __init__(self, model):
        super().__init__(
            tools=[search_quote_history_tool],
            model=model,
            name="quoting_agent",
            description="Fallback pricing agent: estimates per-unit prices for items with no historical quote data, applying the standard two-factor discount formula, and returns a formatted quote line per item."
        )

class SalesAgent(ToolCallingAgent):
    """Agent responsible for handling sales transaction."""

    def __init__(self, model):
        super().__init__(
            tools=[create_transaction_tool, generate_financial_report_tool],
            model=model,
            name="sales_agent",
            description="Process orders, finalise transaction and update system database"
        )

class RequestParserAgent(ToolCallingAgent):
    """Agent responsible for parsing raw customer requests into structured JSON with canonical item names."""

    def __init__(self, model):
        super().__init__(
            tools=[get_all_inventory_tool],
            model=model,
            name="request_parser_agent",
            description="Parses a raw customer request into structured JSON by fuzzy-matching item descriptions to canonical catalog names and extracting quantities. Returns a JSON list where each entry has canonical_name (null if no catalog match), quantity, and customer_description."
        )

def make_orchestrator_tools(inventory_agent, quoting_agent, sales_agent, request_parser_agent, context):
    @tool
    def parse_customer_request(customer_request: str) -> str:
        """
        Use this tool FIRST for every customer request to extract structured item data.

        This tool parses the natural language customer request and returns a JSON list of items
        with canonical catalog names and quantities. Items that cannot be matched to the catalog
        will have canonical_name set to null.

        Always call this before delegate_check_inventory.

        Args:
            customer_request (str): The full customer request text including items and quantities.

        Returns:
            str: A JSON array of items, e.g.:
                 [{"canonical_name": "Glossy paper", "quantity": 200, "customer_description": "A4 glossy paper"},
                  {"canonical_name": null, "quantity": 300, "customer_description": "streamers"}]
        """
        request_date = context.get("request_date", "unknown")
        return request_parser_agent.run(f"""
            The customer request is: "{customer_request}"
            The request date is: {request_date}

            Your job is to extract every item and quantity from the customer request,
            then match each to the canonical catalog name. Follow these steps exactly:

            1. Call get_all_inventory_tool with as_of_date="{request_date}" to get the full catalog.
               The catalog keys are the ONLY valid canonical item names.

            2. For each item mentioned in the customer request:
               a. Identify the item description and quantity requested.
               b. Fuzzy-match the description to the closest catalog key using semantic similarity.
                  Examples of correct matches:
                  - "A4 glossy paper" → "Glossy paper"
                  - "heavy cardstock (white)" → "Cardstock"
                  - "colored paper (assorted)" → "Colored paper"
                  - "A4 white printer paper" → "Printer paper"
                  - "poster boards" → "Poster board"
                  - "reams of printer paper" → "Printer paper"
               c. If there is no reasonable catalog match (e.g. "streamers", "balloons", "tickets",
                  "napkins", "cups", "plates", "washi tape"), set canonical_name to null.

            3. Return ONLY a valid JSON array. No preamble, explanation, or markdown fences. Example:
               [{{"canonical_name": "Glossy paper", "quantity": 200, "customer_description": "A4 glossy paper"}},
                {{"canonical_name": null, "quantity": 300, "customer_description": "streamers"}}]

            Return the JSON array now.
        """)

    @tool
    def delegate_check_inventory(structured_items: str, customer_request: str) -> str:
        """
        Use this tool after parse_customer_request to verify stock availability per item.

        Conditions A and B are evaluated in Python using live catalog data — the InventoryAgent
        is only called for items that actually need a reorder feasibility check (cash + delivery date).

        Always pass the full JSON output from parse_customer_request as structured_items,
        and the original customer request text as customer_request (used for deadline parsing).

        Args:
            structured_items (str): JSON array from parse_customer_request containing
                                    canonical_name, quantity, and customer_description per item.
            customer_request (str): The original customer request text (used to extract delivery deadline).

        Returns:
            str: A per-item report with one of three outcomes — FULFILLABLE, REORDER INITIATED,
                 or CANNOT FULFILL — each with a reason.
        """
        import json
        request_date = context.get("request_date", "unknown")

        # --- Python pre-evaluation: fetch catalog and evaluate both conditions ---
        catalog = get_all_inventory(request_date)

        try:
            items = json.loads(structured_items)
        except (json.JSONDecodeError, TypeError):
            try:
                items = ast.literal_eval(str(structured_items))
            except Exception:
                return "ERROR: Could not parse structured_items JSON from parse_customer_request."

        report_lines = []  # Immediately settled outcomes (FULFILLABLE or null/missing CANNOT FULFILL)
        needs_reorder = [] # Items where Condition A or B triggered → need cash + delivery check

        for item in items:
            canonical_name = item.get("canonical_name")
            quantity = item.get("quantity", 0)
            customer_desc = item.get("customer_description", canonical_name or "unknown item")

            # Null canonical_name means RequestParserAgent found no catalog match
            if not canonical_name:
                report_lines.append(
                    f"- {customer_desc}: CANNOT FULFILL — item not found in catalog."
                )
                continue

            # Canonical name not present in the live catalog (e.g. zero stock, never existed)
            if canonical_name not in catalog:
                report_lines.append(
                    f"- {canonical_name}: CANNOT FULFILL — item not found in live catalog."
                )
                continue

            stock_data = catalog[canonical_name]
            current_stock = stock_data["current_stock"]
            min_stock_level = stock_data["min_stock_level"]

            condition_a = current_stock < min_stock_level  # Stock below reorder threshold
            condition_b = current_stock < quantity         # Insufficient stock for this order

            if not condition_a and not condition_b:
                report_lines.append(
                    f"- {canonical_name}: FULFILLABLE — {current_stock} units in stock, {quantity} requested."
                )
            else:
                needs_reorder.append({
                    "canonical_name": canonical_name,
                    "quantity": quantity,
                    "current_stock": current_stock,
                    "min_stock_level": min_stock_level,
                })

        # If nothing needs reorder, return the Python-computed report directly — no agent call
        if not needs_reorder:
            return "\n".join(report_lines)

        # --- InventoryAgent: only handles reorder feasibility for NEEDS_REORDER items ---
        needs_reorder_text = "\n".join(
            f"  - {r['canonical_name']}: requested={r['quantity']}, current_stock={r['current_stock']}, min_stock_level={r['min_stock_level']}"
            for r in needs_reorder
        )

        agent_report = inventory_agent.run(f"""
            The customer request is: "{customer_request}"
            The request date is: {request_date}

            The following items have been pre-verified by Python as needing a reorder feasibility check
            (either current stock is below the reorder threshold, or insufficient for the order):
            {needs_reorder_text}

            For EACH item above, determine whether it can be reordered in time:

            a. Call get_cash_balance_tool ONCE with as_of_date="{request_date}".
               If cash balance <= 0: assign CANNOT FULFILL (insufficient funds) to ALL items.
               Do NOT call get_supplier_delivery_date_tool. Return your report immediately.

            b. If cash > 0, for each item call get_supplier_delivery_date_tool with
               input_date_str="{request_date}" and the item's quantity.

            c. Extract the customer's delivery deadline from this text:
               "{customer_request}"

            d. For each item, compare delivery date to the customer's deadline:
               - delivery date <= deadline → REORDER INITIATED.
                 State: "Reorder initiated for [canonical_name]: estimated restock by [date]."
               - delivery date > deadline  → CANNOT FULFILL.
                 State: "Restock arrives [date], after customer deadline [deadline]."

            Return exactly ONE line per item in this format:
            - [canonical_name]: REORDER INITIATED / CANNOT FULFILL — [reason]

            Only include the items listed above. Do not add any other items.
        """)

        # Combine Python-settled outcomes with the agent's reorder report
        all_lines = report_lines + [agent_report.strip()]
        return "\n".join(all_lines)

    @tool
    def generate_quote(customer_request: str, inventory_report: str) -> str:
        """
        Use this tool after delegate_check_inventory to generate a price quote.

        Call this tool when:
        - delegate_check_inventory has returned a report and at least one item is FULFILLABLE
          or REORDER INITIATED.
        - You need to generate a competitive, stock-aware price quote for the customer.

        Always pass both the original customer request AND the full inventory report from
        delegate_check_inventory so the quoting agent knows exactly which items to price.

        Args:
            customer_request (str): The original customer request including items, quantities, and date.
            inventory_report (str): The full output from delegate_check_inventory, listing each item's
                                    fulfillment outcome (FULFILLABLE / REORDER INITIATED / CANNOT FULFILL).

        Returns:
            str: A price breakdown per fulfillable item: item name, quantity, base price per unit,
                 discount applied, and final total price.
        """
        import json
        import re
        from collections import Counter
        request_date = context.get("request_date", "unknown")

        # Parse inventory report to identify items to quote
        items_to_quote = []
        for line in inventory_report.strip().split('\n'):
            stripped = line.strip().lstrip('- ')
            if ': FULFILLABLE' in stripped:
                m = re.search(r'^(.+?): FULFILLABLE — (\d+) units in stock, (\d+) requested\.', stripped)
                if m:
                    items_to_quote.append({
                        'name': m.group(1), 'status': 'FULFILLABLE',
                        'current_stock': int(m.group(2)), 'quantity': int(m.group(3))
                    })
            elif ': REORDER INITIATED' in stripped:
                m_name = re.search(r'^(.+?): REORDER INITIATED', stripped)
                m_qty = re.search(r'(\d+) requested', stripped)
                m_stock = re.search(r'(\d+) units in stock', stripped)
                if m_name:
                    name = m_name.group(1)
                    if m_stock:
                        current_stock = int(m_stock.group(1))
                    else:
                        # REORDER INITIATED lines from the agent don't include a stock count,
                        # so use get_stock_level for the precise per-item figure.
                        try:
                            current_stock = get_stock_level(name, request_date)["current_stock"].iloc[0]
                        except Exception:
                            current_stock = 0
                    items_to_quote.append({
                        'name': name, 'status': 'REORDER INITIATED',
                        'current_stock': current_stock,
                        'quantity': int(m_qty.group(1)) if m_qty else 0
                    })

        if not items_to_quote:
            return "No FULFILLABLE or REORDER INITIATED items to quote."

        # Fetch historical base prices in Python — no agent needed for DB lookups
        def _extract_base_price(item_name):
            records = search_quote_history([item_name], limit=5)
            prices = []
            # Use first keyword of item name to anchor the search window
            anchor = item_name.lower().split()[0]
            for record in records:
                if record.get('total_amount', -1) <= 0:
                    continue
                explanation = record.get('quote_explanation', '')
                expl_lower = explanation.lower()
                # Find all occurrences of the item name anchor in the explanation
                start = 0
                while True:
                    idx = expl_lower.find(anchor, start)
                    if idx == -1:
                        break
                    # Look for a price within 200 chars after the item mention
                    window = explanation[idx:idx + 200]
                    pm = re.search(r'\$(\d+\.\d+)\s+(?:each|per\s+\w+)', window)
                    if pm:
                        price = float(pm.group(1))
                        if 0.01 <= price <= 5.0:
                            prices.append(price)
                    start = idx + 1
            if not prices:
                return None
            return Counter(prices).most_common(1)[0][0]

        # Apply two-factor discount formula in Python and return an explicit rationale
        def _compute_discount_and_reason(quantity, current_stock, status):
            if quantity < 500:
                factor_a = 0.0
                tier_reason = "quantity under 500"
            elif quantity <= 1000:
                factor_a = 0.05
                tier_reason = "quantity between 500 and 1000"
            elif quantity <= 5000:
                factor_a = 0.10
                tier_reason = "quantity between 1001 and 5000"
            else:
                factor_a = 0.15
                tier_reason = "quantity above 5000"
            if status == 'REORDER INITIATED' or current_stock < 1.5 * quantity:
                return factor_a * 0.5, f"tight stock or reorder status reduced tier discount; {tier_reason}"
            if current_stock >= 3 * quantity:
                return min(factor_a + 0.05, 0.20), f"abundant stock increased tier discount; {tier_reason}"
            return factor_a, f"standard tier discount applied; {tier_reason}"

        quote_records = []
        unknown_price_items = []
        for item in items_to_quote:
            base_price = _extract_base_price(item['name'])
            if base_price is None:
                unknown_price_items.append(item)
                continue
            discount, discount_reason = _compute_discount_and_reason(
                item['quantity'], item['current_stock'], item['status']
            )
            discount_pct = round(discount * 100, 1)
            final_price = round(base_price * item['quantity'] * (1 - discount), 2)
            quote_records.append(
                {
                    "item_name": item["name"],
                    "transaction_type": "sales",
                    "quantity": int(item["quantity"]),
                    "base_price_per_unit": round(float(base_price), 2),
                    "discount_applied": float(discount_pct),
                    "discount_reason": discount_reason,
                    "final_price": float(final_price),
                    "date": request_date,
                }
            )

        if not unknown_price_items:
            return json.dumps(quote_records)

        # Fallback: call quoting agent only for items with no price history
        fallback_text = "\n".join(
            f"  - {it['name']}: quantity={it['quantity']}, "
            f"current_stock={it['current_stock']}, status={it['status']}"
            for it in unknown_price_items
        )
        known_quote_records = list(quote_records)
        agent_result = quoting_agent.run(
            f"Request date: {request_date}\n"
            f"Estimate per-unit prices for these items (no price history found):\n{fallback_text}\n"
            f"Discount rules:\n"
            f"  Factor A: <500=0%, 500-1000=5%, 1001-5000=10%, >5000=15%\n"
            f"  Factor B: REORDER/tight(stock<1.5xqty)=halve A; abundant(stock>=3xqty)=A+5%(max 20%); comfortable=A\n"
            f"Return ONLY a JSON array. Each object must include keys: "
            f"item_name, transaction_type, quantity, base_price_per_unit, discount_applied, "
            f"discount_reason, final_price, date. "
            f"Call final_answer ONCE. Do NOT call search_quote_history_tool."
        )
        fallback_records = []
        try:
            parsed = json.loads(agent_result)
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                for rec in parsed:
                    if not isinstance(rec, dict):
                        continue
                    item_name = str(rec.get("item_name", "")).strip()
                    if not item_name:
                        continue
                    fallback_records.append(
                        {
                            "item_name": item_name,
                            "transaction_type": "sales",
                            "quantity": int(rec.get("quantity", 0) or 0),
                            "base_price_per_unit": float(rec.get("base_price_per_unit", 0.0) or 0.0),
                            "discount_applied": float(rec.get("discount_applied", 0.0) or 0.0),
                            "discount_reason": str(rec.get("discount_reason", "estimated price due to limited historical data")),
                            "final_price": float(rec.get("final_price", 0.0) or 0.0),
                            "date": str(rec.get("date", request_date)),
                        }
                    )
        except Exception:
            # Keep deterministic behavior even if fallback agent fails format requirements.
            fallback_records = []
        return json.dumps(known_quote_records + fallback_records)

    @tool
    def process_sale(customer_request: str, quote: str) -> str:
        """
        Use this when you need an order to be processed for a customer request when both the stock is verified and the quote is generated.

        Call this tool when:
        - You need to make an update to the database based on the customer's order.
        - Verify if the company has enough cash to cover the order transaction before committing to it.

        Args:
            customer_request (str): The full customer request including items, quantities, and date.
            quote (str): The generated price quote containing item_name, transaction_type, quantity, price, and date for each item.

        Returns:
            str: A confirmation report of all recorded transactions and final order status.
        """
        request_date = context.get("request_date", "unknown")
        return sales_agent.run(f"""
            The generated quote is: "{quote}"
            The request date is: {request_date}

            Please do the following:
            1. The quote is a JSON array. Extract ONLY the items explicitly listed: item_name, transaction_type, quantity, and final_price.
               The quote is the SOLE source of items to record. Do NOT create transactions for any items not present in the quote.
            2. Use {request_date} as the transaction date for ALL transactions. Do not use any other date.
            3. For EACH item in the quote (and ONLY those items), record a transaction using create_transaction_tool with transaction_type="sales" and price=final_price from the quote.
            4. Once all transactions are recorded, call generate_financial_report_tool with as_of_date="{request_date}" to confirm the updated cash balance and inventory value.
            5. Return ONLY transaction confirmation lines for the recorded sale items.
               Do NOT include cash balance, inventory value, total assets, costs, margins, or other internal financial details.
        """)

    @tool
    def compose_customer_response(customer_request: str, inventory_report: str, quote: str) -> str:
        """
        Use this as the final step to produce a customer-facing response.

        This tool composes a deterministic summary from inventory and quote outputs.
        It includes pricing rationale and fulfillment reasons while excluding internal financial data.

        Args:
            customer_request (str): Original customer request text.
            inventory_report (str): Full inventory result from delegate_check_inventory.
            quote (str): Full quote output from generate_quote.

        Returns:
            str: Final customer-facing response with itemized pricing and reasons.
        """
        import json
        import re

        inventory_status = {}
        for line in str(inventory_report).splitlines():
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            m = re.match(r"^- (.+?): (FULFILLABLE|REORDER INITIATED|CANNOT FULFILL)\s+[—-]\s+(.+)$", stripped)
            if not m:
                continue
            inventory_status[m.group(1).strip()] = {
                "status": m.group(2).strip(),
                "reason": m.group(3).strip(),
            }

        quote_items = {}
        try:
            parsed_quote = json.loads(str(quote))
            if isinstance(parsed_quote, dict):
                parsed_quote = [parsed_quote]
            if isinstance(parsed_quote, list):
                for rec in parsed_quote:
                    if not isinstance(rec, dict):
                        continue
                    name = str(rec.get("item_name", "")).strip()
                    if not name:
                        continue
                    quote_items[name] = {
                        "quantity": int(rec.get("quantity", 0) or 0),
                        "base_price_per_unit": float(rec.get("base_price_per_unit", 0.0) or 0.0),
                        "discount_applied": float(rec.get("discount_applied", 0.0) or 0.0),
                        "final_price": float(rec.get("final_price", 0.0) or 0.0),
                        "discount_reason": str(
                            rec.get("discount_reason", "discount applied per pricing policy")
                        ).strip(),
                        "date": str(rec.get("date", "")),
                    }
        except Exception:
            quote_items = {}

        fulfilled_lines = []
        unavailable_lines = []
        for item_name, status_data in inventory_status.items():
            status = status_data["status"]
            reason = status_data["reason"]
            quote_data = quote_items.get(item_name)
            if status in {"FULFILLABLE", "REORDER INITIATED"} and quote_data:
                fulfilled_lines.append(
                    f"- {item_name}: {quote_data['quantity']} units, total ${quote_data['final_price']:.2f} "
                    f"(base ${quote_data['base_price_per_unit']:.2f}/unit, discount {quote_data['discount_applied']:.1f}% "
                    f"because {quote_data['discount_reason']}). Status: {status}. {reason}"
                )
            elif status == "CANNOT FULFILL":
                unavailable_lines.append(f"- {item_name}: {reason}")

        if not fulfilled_lines and unavailable_lines:
            return (
                "Order Update:\n"
                "Unavailable Items and Reasons:\n"
                + "\n".join(unavailable_lines)
            )

        response_parts = ["Order Update:"]
        if fulfilled_lines:
            response_parts.append("Fulfilled or Reordered Items:")
            response_parts.extend(fulfilled_lines)
        if unavailable_lines:
            response_parts.append("Unavailable Items and Reasons:")
            response_parts.extend(unavailable_lines)
        response_text = "\n".join(response_parts).strip()

        # Never return a header-only response; provide a deterministic fallback.
        if response_text == "Order Update:":
            return (
                "Order Update:\n"
                "We could not produce a fully itemized customer summary for this request.\n"
                "Inventory Assessment:\n"
                f"{str(inventory_report).strip() or 'No inventory assessment available.'}"
            )
        return response_text

    return [parse_customer_request, delegate_check_inventory, generate_quote, process_sale, compose_customer_response]


def sanitize_customer_response(response_text: str) -> str:
    """
    Remove internal errors/diagnostics from customer-facing messages.

    Args:
        response_text (str): Candidate customer response.

    Returns:
        str: Safe customer-facing response.
    """
    text = str(response_text or "").strip()
    if not text:
        return (
            "Order Update:\n"
            "We could not complete a full automated summary for this request.\n"
            "Please confirm item names, quantities, and delivery date so we can proceed."
        )

    lowered = text.lower()
    blocked_markers = [
        "error:",
        "traceback",
        "exception",
        "could not parse structured_items",
        "jsondecodeerror",
        "valueerror",
        "keyerror",
        "typeerror",
        "internal",
    ]
    if any(marker in lowered for marker in blocked_markers):
        return (
            "Order Update:\n"
            "We could not fully interpret part of your request.\n"
            "Please rephrase item names and quantities (for example: '500 sheets of A4 paper') and resend."
        )

    forbidden_terms = [
        "cash balance",
        "inventory value",
        "total assets",
        "profit margin",
        "internal financial",
    ]
    kept_lines = []
    for line in text.splitlines():
        if any(term in line.lower() for term in forbidden_terms):
            continue
        kept_lines.append(line)
    safe_text = "\n".join(kept_lines).strip()
    return safe_text or (
        "Order Update:\n"
        "Your request was processed, but detailed output is temporarily unavailable."
    )

class Orchestrator(ToolCallingAgent):
    """Orchestrator that coordinates the Beaver's Choice Paper Company system."""

    def __init__(self, model):
        self._context = {"request_date": None}
        self.request_parser_agent = RequestParserAgent(model)
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.sales_agent = SalesAgent(model)

        tools = make_orchestrator_tools(
            self.inventory_agent,
            self.quoting_agent,
            self.sales_agent,
            self.request_parser_agent,
            self._context
        )

        super().__init__(
            tools=tools,
            model=model,
            name="orchestrator",
            description="""
            You are the orchestrator for The Beaver's Choice Paper Company.
            You coordinate between the request parser, inventory agent, quoting agent, and sales agent.

            For every customer request, follow this exact workflow in order:

            1. Call parse_customer_request with the full customer request text.
               It returns a JSON list of items: each has canonical_name (null if not in catalog),
               quantity, and customer_description.

            2. Review the parsed items:
               - If ALL items have canonical_name null, inform the customer that none of the
                 requested items are available in the catalog, and stop. Do not call any other tool.
               - Otherwise, proceed to step 3.

            3. Call delegate_check_inventory, passing:
               - structured_items: the full JSON output from step 1.
               - customer_request: the original customer request text.
               It returns a per-item report: FULFILLABLE, REORDER INITIATED, or CANNOT FULFILL.

            4. Review the inventory report:
               - If ALL items are CANNOT FULFILL, inform the customer and stop. Do not call generate_quote.
               - Otherwise, proceed to step 5.

            5. Call generate_quote, passing BOTH the original customer_request AND the full inventory
               report from step 3 as inventory_report. The quoting agent will price only fulfillable items.

            6. If the quote from step 5 contains at least one priced item, you MUST call process_sale
               with the customer_request and the generated quote to record all transactions and update
               the financial report. Do NOT skip this step or call final_answer before it, even if
               some items in the order were CANNOT FULFILL.

            7. After step 6, call compose_customer_response with:
               - customer_request (original request text)
               - inventory_report (full output from step 3)
               - quote (full output from step 5)

            8. Return the compose_customer_response output VERBATIM as the final answer.
               Do not rewrite, paraphrase, summarize, or replace it with your own text.
               The final answer must include itemized base price, discount percentage, discount reason,
               final price for quoted items, and cannot include internal financial data.
            """
        )

    def run(self, task, **kwargs):
        import re

        match = re.search(r'\(Date of request: (\d{4}-\d{2}-\d{2})\)', task)
        if match:
            self._context["request_date"] = match.group(1)
        response = super().run(task, **kwargs)
        return sanitize_customer_response(response)

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE

    orchestrator = Orchestrator(model)

    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST

        # Persist exactly what the customer should see, after safety sanitization.
        response = sanitize_customer_response(orchestrator.run(request_with_date))

        ############
        ############
        ############

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results

if __name__ == "__main__":
    results = run_test_scenarios()
