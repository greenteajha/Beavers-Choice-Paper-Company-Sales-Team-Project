# Beaver's Choice Paper Company Sales System

## Overview

This project implements a multi-agent sales workflow for a paper supplier.  
The system processes customer requests, checks inventory constraints, generates pricing, records transactions, and returns a customer-facing response with fulfillment details.

## What The App Does

- Parses free-text customer requests into structured item and quantity data.
- Checks whether each requested item is:
- `FULFILLABLE`
- `REORDER INITIATED`
- `CANNOT FULFILL`
- Produces structured pricing with:
- base unit price
- discount percentage
- discount rationale
- final total per item
- Records confirmed sales transactions in the database.
- Generates a sanitized final response suitable for customer communication.

## Architecture

The orchestrator uses tool-calling delegation to coordinate the workflow:

1. `parse_customer_request`
2. `delegate_check_inventory`
3. `generate_quote`
4. `process_sale` (only when priced items exist)
5. `compose_customer_response`

The sequencing is directed by the orchestrator agent prompt and executed through registered tools, with worker-agent calls encapsulated inside those tools.

Worker agents are specialized by function:

- `RequestParserAgent` for extraction and catalog mapping
- `InventoryAgent` for reorder feasibility
- `QuotingAgent` for fallback pricing when no history exists
- `SalesAgent` for transaction recording and internal finance snapshot

## Project Files

- `project_starter.py`: main implementation
- `munder_difflin.db`: SQLite database
- `quote_requests.csv`: request dataset
- `quote_requests_sample.csv`: evaluation sample input
- `quotes.csv`: historical quote records
- `test_results.csv`: evaluation output
- `test_run_results.txt`: verbose execution logs
- `reflection_report.txt`: architecture and evaluation reflection

## Setup

1. Ensure Python 3.8+ is installed.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` with required key:

```bash
UDACITY_API_KEY=your_key_here
```

## Run

```bash
python project_starter.py
```

## Output

A run will:

- print per-request processing logs
- update financial and inventory state
- write final records to `test_results.csv`

## Notes

- Customer responses are sanitized to avoid leaking internal diagnostics or finance-only details.
- Pricing and discount computation are primarily handled in Python for consistency, with fallback agent assistance when needed.
