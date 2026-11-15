
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# NVD API configuration
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "nvd"

API_KEY = os.getenv("NVD_API_KEY")

HEADERS = {}
if API_KEY:
    HEADERS["apiKey"] = API_KEY

MAX_RETRIES = 5

RETRYABLE_STATUS_CODES = {
    429, 500, 502, 503, 504
}


def save_cves(cves):
    """Save collected CVE records as a JSON file."""

    if not cves:
        print("\nNo CVE records collected.")
        print("No empty dataset will be saved.")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = OUTPUT_DIR / f"nvd_cves_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            cves,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nDataset saved successfully: {output_file}")

    return output_file


def make_api_request(params):
    """Send an NVD request with automatic retries."""

    for attempt in range(MAX_RETRIES):

        try:
            response = requests.get(
                NVD_API_URL,
                params=params,
                headers=HEADERS,
                timeout=60
            )

            # Retry temporary server errors and rate limits.
            if response.status_code in RETRYABLE_STATUS_CODES:

                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")

                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    wait_time = min(2 ** (attempt + 1), 60)

                print(
                    f"NVD returned HTTP {response.status_code}. "
                    f"Retrying in {wait_time} seconds "
                    f"({attempt + 1}/{MAX_RETRIES})..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as error:

            print(f"HTTP error: {error}")
            raise

        except requests.exceptions.RequestException as error:

            if attempt == MAX_RETRIES - 1:
                print(
                    f"Request failed after {MAX_RETRIES} attempts: "
                    f"{error}"
                )
                raise

            wait_time = min(2 ** (attempt + 1), 60)

            print(f"Connection error: {error}")
            print(f"Retrying in {wait_time} seconds...")

            time.sleep(wait_time)

        except ValueError as error:

            print(f"Invalid JSON response from NVD: {error}")
            raise

    return None


def extract_cve_record(cve):
    """Extract useful fields from an NVD CVE record."""

    descriptions = cve.get("descriptions", [])

    english_description = next(
        (
            item.get("value")
            for item in descriptions
            if item.get("lang") == "en"
        ),
        "No English description available"
    )

    metrics = cve.get("metrics", {})

    cvss_score = None
    severity = "UNKNOWN"

    metric_types = [
        "cvssMetricV40",
        "cvssMetricV31",
        "cvssMetricV30",
        "cvssMetricV2"
    ]

    for metric_type in metric_types:

        metric_list = metrics.get(metric_type, [])

        if metric_list:

            metric = metric_list[0]
            cvss_data = metric.get("cvssData", {})

            cvss_score = cvss_data.get("baseScore")

            severity = (
                cvss_data.get("baseSeverity")
                or metric.get("baseSeverity")
                or "UNKNOWN"
            )

            break

    weaknesses = []

    for weakness in cve.get("weaknesses", []):

        for description in weakness.get("description", []):

            if description.get("lang") == "en":
                weaknesses.append(
                    description.get("value")
                )

    references = [
        reference.get("url")
        for reference in cve.get("references", [])
        if reference.get("url")
    ]

    return {
        "cve_id": cve.get("id"),
        "source": "NVD",
        "published": cve.get("published"),
        "last_modified": cve.get("lastModified"),
        "description": english_description,
        "cvss_score": cvss_score,
        "severity": severity,
        "weaknesses": weaknesses,
        "references": references
    }


def fetch_cves(keyword=None, max_records=100):
    """
    Fetch CVE records from the official NVD API.

    Supports keyword search and configurable record limits.
    """

    if max_records <= 0:
        raise ValueError("max_records must be greater than zero.")

    all_cves = []
    start_index = 0

    print("\n================================")
    print("NVD CVE DATA COLLECTION")
    print("================================")

    print(f"Maximum records: {max_records}")

    if keyword:
        print(f"Keyword: {keyword}")

    while len(all_cves) < max_records:

        remaining = max_records - len(all_cves)
        page_size = min(remaining, 2000)

        params = {
            "startIndex": start_index,
            "resultsPerPage": page_size
        }

        if keyword:
            params["keywordSearch"] = keyword

        print(f"\nFetching records from index {start_index}...")

        try:
            data = make_api_request(params)

        except (
            requests.exceptions.RequestException,
            ValueError
        ):

            print("\nUnable to retrieve the next page.")

            if all_cves:
                print("Saving previously collected records.")
                save_cves(all_cves)
            else:
                print("No records were collected.")

            return all_cves

        vulnerabilities = data.get("vulnerabilities", [])

        if not vulnerabilities:
            print("No more CVE records available.")
            break

        for item in vulnerabilities:

            cve = item.get("cve", {})

            record = extract_cve_record(cve)

            all_cves.append(record)

        print(
            f"Successfully collected "
            f"{len(all_cves)} / {max_records} records."
        )

        start_index += len(vulnerabilities)

        total_results = data.get("totalResults", 0)

        if start_index >= total_results:
            break

        if len(all_cves) < max_records:
            time.sleep(6 if API_KEY else 7)

    save_cves(all_cves)

    print("\n================================")
    print("COLLECTION SUMMARY")
    print("================================")
    print(f"Total CVEs collected: {len(all_cves)}")
    print("Status: Completed")

    return all_cves


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download CVE records from NVD"
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Optional CVE keyword search"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of CVE records"
    )

    args = parser.parse_args()

    fetch_cves(
        keyword=args.keyword,
        max_records=args.limit
    )
