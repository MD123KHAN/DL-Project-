
from pathlib import Path
from difflib import SequenceMatcher
from functools import lru_cache
import re

import pandas as pd


# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "model_files"
    / "cars_dataset_100000_india_global.csv"
)

_df = None


# --------------------------------------------------
# Text helpers
# --------------------------------------------------

def normalize_text(value):
    """Normalize text and treat hyphens as spaces."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def value_or_na(value):
    """Convert missing values into readable strings."""
    if value is None or pd.isna(value):
        return "Not available"

    if isinstance(value, float) and value.is_integer():
        value = int(value)

    value = str(value).strip()

    if value.casefold() in {"", "nan", "none", "null"}:
        return "Not available"

    return value


def clean_query(query):
    """Remove common search filler words."""
    text = normalize_text(query)

    ignored = {
        "car", "cars", "vehicle", "vehicles",
        "please", "find", "search", "show",
        "specification", "specifications",
        "specs", "details", "information", "features"
    }

    return " ".join(
        token for token in text.split()
        if token not in ignored
    )


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------

def load_dataset():
    """Load the CSV once and prepare normalized search columns."""
    global _df

    if _df is not None:
        return _df

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Car dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    required = ["Make", "Model", "Year"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for column in ["Make", "Model", "Trim_Variant"]:
        if column not in df.columns:
            df[column] = ""

        df[column] = df[column].fillna("").astype(str).str.strip()

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    df["_make_norm"] = df["Make"].map(normalize_text)
    df["_model_norm"] = df["Model"].map(normalize_text)
    df["_variant_norm"] = df["Trim_Variant"].map(normalize_text)

    df["_search_text"] = (
        df["_make_norm"] + " "
        + df["_model_norm"] + " "
        + df["_variant_norm"]
    ).str.strip()

    _df = df
    return _df


# --------------------------------------------------
# Fuzzy matching helpers
# --------------------------------------------------

def token_matches(query_token, available_tokens):
    """
    Match exact words, prefixes, and likely misspellings.
    Example: scropa can match scorpio.
    """
    for word in available_tokens:
        if word == query_token:
            return True

        # Allow typing an incomplete word.
        if len(query_token) >= 2 and word.startswith(query_token):
            return True

        # Handle small spelling mistakes.
        if len(query_token) >= 4:
            similarity = SequenceMatcher(
                None, query_token, word
            ).ratio()

            if similarity >= 0.67:
                return True

    return False


def score_vehicle(query, row):
    """Score a vehicle against a partial or misspelled query."""
    query_tokens = query.split()

    if not query_tokens:
        return 0

    make_tokens = row["_make_norm"].split()
    model_tokens = row["_model_norm"].split()
    variant_tokens = row["_variant_norm"].split()

    all_tokens = make_tokens + model_tokens + variant_tokens

    score = 0

    for token in query_tokens:
        if not token_matches(token, all_tokens):
            return 0

        if token_matches(token, make_tokens):
            score += 10

        if token_matches(token, model_tokens):
            score += 12

        if token_matches(token, variant_tokens):
            score += 6

    # Prefer exact matches over fuzzy matches.
    full_name = row["_search_text"]

    if query == full_name:
        score += 50
    elif full_name.startswith(query):
        score += 20

    return score


# --------------------------------------------------
# Autocomplete suggestions
# --------------------------------------------------

def get_car_suggestions(query, limit=10):
    """
    Return matching vehicle suggestions while the user types.

    Returns a list of dictionaries containing name, variant and year.
    """
    query = clean_query(query)
    if len(query) < 2:
        return []

    try:
        limit = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        limit = 10

    return _get_car_suggestions_cached(query, limit)


@lru_cache(maxsize=256)
def _get_car_suggestions_cached(query, limit):
    """Cache normalized searches and score only a narrowed candidate set."""
    df = load_dataset()

    scored = []
    candidate_rows = df
    query_tokens = query.split()
    if query_tokens:
        mask = pd.Series(True, index=df.index)
        search_text = df["_search_text"]
        for token in query_tokens:
            mask &= search_text.str.contains(
                rf"\b{re.escape(token)}",
                regex=True,
                na=False,
            )
        narrowed = df.loc[mask]
        # Keep typo-tolerant matching when no exact/prefix candidate exists.
        if not narrowed.empty:
            candidate_rows = narrowed

    for index, row in candidate_rows.iterrows():
        score = score_vehicle(query, row)

        if score > 0:
            scored.append((score, index))

    if not scored:
        return []

    scored.sort(
        key=lambda item: (
            -item[0],
            -(df.at[item[1], "Year"]
              if pd.notna(df.at[item[1], "Year"]) else 0)
        )
    )

    suggestions = []
    seen = set()

    for score, index in scored:
        row = df.loc[index]

        make = value_or_na(row["Make"])
        model = value_or_na(row["Model"])
        variant = value_or_na(row["Trim_Variant"])

        year = (
            str(int(row["Year"]))
            if pd.notna(row["Year"])
            else "Year unavailable"
        )

        name = f"{make} {model}".strip()

        label_parts = [name]

        if variant != "Not available":
            label_parts.append(variant)

        label_parts.append(year)

        label = " ".join(label_parts)

        key = normalize_text(label)

        if key in seen:
            continue

        seen.add(key)

        suggestions.append({
            "label": label,
            "name": name,
            "manufacturer": make,
            "model": model,
            "variant": variant,
            "year": year
        })

        if len(suggestions) >= limit:
            break

    return suggestions


# --------------------------------------------------
# Search a selected car
# --------------------------------------------------

def find_car(query):
    """Find the best matching car record in the dataset."""
    df = load_dataset()
    query = clean_query(query)

    if not query:
        return None

    year_match = re.search(r"\b(?:19|20)\d{2}\b", query)
    requested_year = int(year_match.group()) if year_match else None

    search_query = query

    if year_match:
        search_query = (
            query[:year_match.start()]
            + " "
            + query[year_match.end():]
        )

    search_query = " ".join(search_query.split())

    candidates = df

    if requested_year is not None:
        candidates = candidates[
            candidates["Year"] == requested_year
        ]

    if candidates.empty:
        return None

    # Narrow ordinary exact/prefix searches with vectorized string matching
    # before the slower fuzzy scorer. This avoids scoring every row in the
    # 100k-record dataset for common queries such as "Mahindra Thar Roxx".
    candidate_rows = candidates
    query_tokens = search_query.split()
    if query_tokens:
        mask = pd.Series(True, index=candidates.index)
        search_text = candidates["_search_text"]
        for token in query_tokens:
            mask &= search_text.str.contains(
                rf"\b{re.escape(token)}",
                regex=True,
                na=False,
            )
        narrowed = candidates.loc[mask]
        if not narrowed.empty:
            candidate_rows = narrowed

    scored = []

    for index, row in candidate_rows.iterrows():
        score = score_vehicle(search_query, row)

        if score > 0:
            scored.append((score, index))

    if not scored:
        return None

    scored.sort(
        key=lambda item: (
            -item[0],
            -(df.at[item[1], "Year"]
              if pd.notna(df.at[item[1], "Year"]) else 0)
        )
    )

    return df.loc[scored[0][1]].drop(
        labels=[
            "_make_norm", "_model_norm",
            "_variant_norm", "_search_text"
        ],
        errors="ignore"
    )


# --------------------------------------------------
# Format result for the frontend
# --------------------------------------------------

def format_car(row):
    """Convert a Pandas Series or dictionary into frontend JSON data."""

    if row is None:
        return None

    # Support both Pandas Series and ordinary dictionaries.
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    if not isinstance(row, dict):
        raise TypeError(
            "format_car() expects a Pandas Series or dictionary."
        )

    def get_value(column):
        """Safely retrieve and normalize a field."""
        return value_or_na(row.get(column))

    manufacturer = get_value("Make")
    model = get_value("Model")
    variant = get_value("Trim_Variant")
    year = get_value("Year")

    if year != "Not available":
        try:
            year = str(int(float(year)))
        except (ValueError, TypeError, OverflowError):
            pass

    name = " ".join(
        part
        for part in (manufacturer, model)
        if part != "Not available"
    ).strip()

    if not name:
        name = "Unknown vehicle"

    # Find an image URL if one exists in the dataset.
    image_url = ""

    for column in (
        "Car_Image_URL",
        "image_url",
        "Image_URL",
        "Image",
        "Photo_URL",
    ):
        value = get_value(column)

        if value.startswith(("https://", "http://")):
            image_url = value
            break

    formatted_price = get_value("Formatted_Price")
    if formatted_price != "Not available":
        price_label, price_value = "Ex-showroom price", formatted_price
    else:
        price_label, price_value = (
            "Ex-showroom price (INR)",
            get_value("Ex_Showroom_Price_INR"),
        )

    fields = [
        ("Manufacturer", manufacturer),
        ("Model", model),
        ("Year", year),
        ("Variant", variant),
        ("Color", get_value("Color")),
        ("Body style", get_value("Body_Style")),
        ("Doors", get_value("Doors")),
        ("Seating capacity", get_value("Seating_Capacity")),
        ("Drivetrain", get_value("Drivetrain")),
        ("Transmission", get_value("Transmission")),
        ("Fuel type", get_value("Fuel_Type")),
        ("Engine displacement", get_value("Displacement")),
        ("Cylinders", get_value("Cylinders")),
        ("Power", get_value("Power_BHP")),
        ("City mileage", get_value("Mileage_City")),
        ("Highway mileage", get_value("Mileage_Highway")),
        (price_label, price_value),
    ]

    specifications = [
        {"label": label, "value": value}
        for label, value in fields
    ]

    return {
        "name": name,
        "manufacturer": manufacturer,
        "model": model,
        "year": year,
        "variant": variant,
        "description": (
            f"{name} ({year}) is listed in the AutoNex vehicle dataset. "
            "Specifications may vary by year, market and variant."
        ),
        "source_type": "AutoNex vehicle dataset",
        "source_url": "",
        "image_url": image_url,
        "specifications": specifications,
    }
