
"""
AutoNex-AI Flask API
Connects the vehicle dataset, trained TinyGPT SLM,
Wikimedia Commons image search, and frontend.
"""

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import logging
import re

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

from model import find_car, load_dataset
from slm import load_slm, generate_text


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoNex-AI")

app.config["JSON_SORT_KEYS"] = False

# Set to your actual frontend origin if you want to restrict CORS.
# Example: http://127.0.0.1:5500
# Leave unrestricted during local development.
SLM_READY = False
SLM_ERROR = None

# Local car images for fast, reliable demos
IMAGE_DIR = BASE_DIR / "image"

@app.get("/api/local-car-image/<path:filename>")
def get_local_car_image(filename):
    # Only serve image files from backend/image/
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if Path(filename).name != filename:
        abort(404)

    if Path(filename).suffix.lower() not in allowed_extensions:
        abort(404)

    image_path = IMAGE_DIR / filename

    if not image_path.is_file():
        abort(404)

    return send_from_directory(IMAGE_DIR, filename)


# --------------------------------------------------
# Load resources
# --------------------------------------------------

try:
    # Verify that the vehicle dataset is accessible.
    df = load_dataset()
    logger.info("Car dataset loaded: %s rows", len(df))
except Exception:
    logger.exception("Could not load the vehicle dataset.")

try:
    load_slm()
    SLM_READY = True
    logger.info("AutoNex SLM loaded successfully.")
except Exception as exc:
    SLM_ERROR = str(exc)
    logger.exception("Could not load the AutoNex SLM.")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def error_response(message, status_code=400, **extra):
    """Return a consistent JSON error response."""
    payload = {
        "success": False,
        "error": message,
    }
    payload.update(extra)
    return jsonify(payload), status_code


def normalize_car_result(car):
    """Convert a car result into JSON-safe values."""
    if car is None:
        return None

    if hasattr(car, "to_dict"):
        car = car.to_dict()

    if not isinstance(car, dict):
        raise TypeError(
            "find_car() must return a dictionary, pandas Series, or None."
        )

    result = {}

    for key, value in car.items():
        if value is None:
            result[str(key)] = None
            continue

        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass

        if isinstance(value, float):
            import math
            if not math.isfinite(value):
                value = None

        result[str(key)] = value

    return result


def valid_http_url(value):
    """Accept only ordinary HTTP/HTTPS URLs."""
    if not isinstance(value, str):
        return ""

    value = value.strip()

    if not value:
        return ""

    from urllib.parse import urlparse

    try:
        parsed = urlparse(value)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return value
    except ValueError:
        pass

    return ""


def make_car_image_url(car):
    """Find an image URL supplied by the dataset."""
    for key in (
        "image_url",
        "Car_Image_URL",
        "image",
        "photo_url",
        "image_link",
        "photo",
    ):
        url = valid_http_url(car.get(key))
        if url:
            return url

    return ""


def parse_car_query(query):
    """
    Extract a requested year and trim from a query.

    Examples:
        Mahindra Scorpio N Z4 2026
        Mahindra Scorpio N Z8L 2025
        Maruti Suzuki Swift 2024
    """
    query = re.sub(r"\s+", " ", query).strip()

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query)
    year = int(year_match.group(1)) if year_match else None

    if year_match:
        query = (
            query[:year_match.start()] + " " +
            query[year_match.end():]
        )

    query = re.sub(r"\s+", " ", query).strip()

    return query, year


def get_image_metadata_from_car(car):
    """Read optional image attribution fields from a car result."""
    return {
        "image_url": make_car_image_url(car),
        "image_source_page": valid_http_url(
            car.get("image_source_page", "")
        ),
        "image_license": str(car.get("image_license", "") or ""),
    }


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "application": "AutoNex-AI",
        "status": "running",
        "slm_loaded": SLM_READY,
        "endpoints": {
            "health": "/api/health",
            "car_search": "/api/car?q=Mahindra%20Thar",
            "car_image": "/api/car-image?q=Mahindra%20Thar",
            "chat": "/api/chat",
        },
    })


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/api/health")
def health():
    dataset_ready = False
    dataset_rows = 0

    try:
        df = load_dataset()
        dataset_ready = True
        dataset_rows = len(df)
    except Exception:
        logger.exception("Dataset health check failed.")

    return jsonify({
        "success": dataset_ready,
        "status": "ok" if dataset_ready else "degraded",
        "dataset_loaded": dataset_ready,
        "dataset_rows": dataset_rows,
        "slm_loaded": SLM_READY,
        "slm_error": SLM_ERROR,
    }), 200 if dataset_ready else 503


# --------------------------------------------------
# Car search endpoint
# --------------------------------------------------

@app.get("/api/car")
def get_car():
    raw_query = request.args.get("q", "").strip()

    if not raw_query:
        return error_response(
            "Enter a car manufacturer or model name."
        )

    if len(raw_query) > 150:
        return error_response("Search query is too long.")

    try:
        query, requested_year = parse_car_query(raw_query)

        # The existing model.py returns the newest matching record.
        # If the user supplied a year, use the year-aware search
        # implemented below when the dataset supports it.
        car = find_car(query)

        if car is None:
            # Retry with the complete query in case the year or trim
            # is included in the dataset's searchable text.
            car = find_car(raw_query)

        if car is None:
            return error_response(
                f"No matching vehicle found for '{raw_query}'.",
                404,
                query=raw_query,
            )

        car = normalize_car_result(car)

        if car.get("ambiguous"):
            return jsonify({
                "success": False,
                "ambiguous": True,
                "error": "More than one model matches. Please refine your search.",
                "matches": car.get("matches", []),
                "query": raw_query,
            }), 409

        # Preserve the expected response shape for script.js.
        image_data = get_image_metadata_from_car(car)

        return jsonify({
            "success": True,
            "query": raw_query,
            "car": {
                **car,
                **image_data,
            },
            "image_url": image_data["image_url"],
            "source": "dataset",
        }), 200

    except FileNotFoundError:
        logger.exception("Vehicle dataset was not found.")
        return error_response(
            "Vehicle dataset not found. Check the dataset path in model.py.",
            500,
        )

    except Exception:
        logger.exception("Car search failed: %s", raw_query)
        return error_response(
            "Car search failed. Check the backend terminal.",
            500,
        )


# --------------------------------------------------
# Wikimedia Commons vehicle image endpoint
# --------------------------------------------------

@app.get("/api/car-image")
def get_car_image():
    query = request.args.get("q", "").strip()

    if not query:
        return error_response("A car name is required.")

    if len(query) > 150:
        return error_response("Car image search query is too long.")

    # Search Wikimedia Commons file pages.
    # Results can contain irrelevant files, so review the source page.
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": 10,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1000,
        "format": "json",
        "formatversion": 2,
    }

    api_url = (
        "https://commons.wikimedia.org/w/api.php?"
        + urlencode(params)
    )

    req = Request(
        api_url,
        headers={
            "User-Agent": (
                "AutoNexAI/1.0 "
                "(vehicle information demo; contact: developer)"
            ),
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(req, timeout=15) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        pages = result.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())

        candidates = []

        for page in pages:
            title = page.get("title", "")
            info_list = page.get("imageinfo", [])

            if not info_list:
                continue

            info = info_list[0]
            metadata = info.get("extmetadata", {})

            image_url = valid_http_url(
                info.get("thumburl") or info.get("url", "")
            )
            source_url = valid_http_url(
                info.get("descriptionurl", "")
            )

            if not image_url or not source_url:
                continue

            license_name = metadata.get(
                "LicenseShortName", {}
            ).get("value", "Check source page")

            artist = metadata.get(
                "Artist", {}
            ).get("value", "See source page")

            candidates.append({
                "success": True,
                "title": title,
                "image_url": image_url,
                "image_source_page": source_url,
                "image_license": license_name,
                "artist": artist,
                "source": "Wikimedia Commons",
            })

        if not candidates:
            return jsonify({
                "success": False,
                "image_url": "",
                "image_source_page": "",
                "image_license": "",
                "message": "No image was found. Try a more specific model name.",
            }), 404

        # Return the first result; the frontend should handle image
        # loading errors and allow another search if needed.
        return jsonify(candidates[0]), 200

    except HTTPError as exc:
        logger.warning("Wikimedia HTTP error: %s", exc.code)
        return error_response(
            "Wikimedia Commons could not complete the image search.",
            502,
        )

    except (URLError, TimeoutError):
        logger.exception("Wikimedia Commons request failed.")
        return error_response(
            "Image search timed out or could not connect.",
            502,
        )

    except (json.JSONDecodeError, ValueError):
        logger.exception("Invalid Wikimedia response.")
        return error_response(
            "Wikimedia Commons returned an invalid response.",
            502,
        )

    except Exception:
        logger.exception("Unexpected image search failure.")
        return error_response(
            "Image search is temporarily unavailable.",
            502,
        )


# --------------------------------------------------
# AI CAR COMPARISON API
# --------------------------------------------------

def _clean_value(value, default="Not available"):
    """Convert pandas/numpy values to readable JSON-safe text."""
    if value is None:
        return default
    try:
        import pandas as pd
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "n/a"}:
        return default
    return text


def _number(value):
    """Extract the first numeric value from a specification string."""
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def comparison_car_facts(car):
    """Normalize a pandas Series or dict into frontend-friendly car facts."""
    if car is None:
        return None
    if hasattr(car, "to_dict"):
        car = car.to_dict()
    if not isinstance(car, dict):
        raise TypeError("Vehicle search result must be a dictionary or pandas Series.")

    # Normalize the row first; never use a Series in a boolean expression.
    name = " ".join(
        part for part in (
            _clean_value(car.get("Make"), ""),
            _clean_value(car.get("Model"), ""),
        ) if part
    ).strip() or _clean_value(car.get("name") or car.get("model"), "Vehicle")
    variant = _clean_value(car.get("Trim_Variant"), "")
    if variant and variant.casefold() not in name.casefold():
        name = f"{name} {variant}".strip()

    # Keep existing direct dataset fields as simple strings for script.js.
    excluded = {
        "Car_Image_URL", "image_url", "Image_URL", "Image", "Photo_URL",
        "image", "image_source_page", "image_license", "image_attribution",
        "source_url", "description", "_make_norm", "_model_norm",
        "_variant_norm", "_search_text",
    }
    specs = {}
    for key, value in car.items():
        if key in excluded or str(key).startswith("_"):
            continue
        cleaned = _clean_value(value, "")
        if cleaned:
            specs[str(key)] = cleaned

    # If a caller already supplied frontend-style specification objects, flatten them.
    provided = car.get("specifications")
    if isinstance(provided, list):
        for item in provided:
            if isinstance(item, dict) and item.get("label"):
                specs[str(item["label"])] = _clean_value(item.get("value"))
    elif isinstance(provided, dict):
        for key, value in provided.items():
            specs[str(key)] = _clean_value(value)

    return {"name": name, "specifications": specs}


def _spec(specs, *keys):
    """Case-insensitive lookup for common dataset column aliases."""
    lookup = {str(k).casefold().replace(" ", "_"): v for k, v in specs.items()}
    for key in keys:
        normalized = key.casefold().replace(" ", "_")
        if normalized in lookup and _clean_value(lookup[normalized], ""):
            return _clean_value(lookup[normalized])
    return "Not available"


def build_comparison_summary(car1, car2, priority):
    """Produce a short, grounded 2–3 sentence comparison from dataset facts."""
    s1, s2 = car1["specifications"], car2["specifications"]
    n1, n2 = car1["name"], car2["name"]

    price1 = _spec(s1, "Formatted_Price", "Ex_Showroom_Price_INR", "Price", "price")
    price2 = _spec(s2, "Formatted_Price", "Ex_Showroom_Price_INR", "Price", "price")
    power1 = _spec(s1, "Power_BHP", "Power", "BHP", "Horsepower")
    power2 = _spec(s2, "Power_BHP", "Power", "BHP", "Horsepower")
    mileage1 = _spec(s1, "Mileage_City", "City mileage", "City_Mileage")
    mileage2 = _spec(s2, "Mileage_City", "City mileage", "City_Mileage")
    highway1 = _spec(s1, "Mileage_Highway", "Highway mileage", "Highway_Mileage")
    highway2 = _spec(s2, "Mileage_Highway", "Highway mileage", "Highway_Mileage")
    seats1 = _spec(s1, "Seating_Capacity", "Seating capacity", "Seats")
    seats2 = _spec(s2, "Seating_Capacity", "Seating capacity", "Seats")
    drive1 = _spec(s1, "Drivetrain", "Drive_Type", "Drive")
    drive2 = _spec(s2, "Drivetrain", "Drive_Type", "Drive")
    trans1 = _spec(s1, "Transmission", "Transmission_Type")
    trans2 = _spec(s2, "Transmission", "Transmission_Type")
    body1 = _spec(s1, "Body_Style", "Body style", "Body_Type")
    body2 = _spec(s2, "Body_Style", "Body style", "Body_Type")
    fuel1 = _spec(s1, "Fuel_Type", "Fuel type")
    fuel2 = _spec(s2, "Fuel_Type", "Fuel type")

    priority = str(priority or "family").casefold()
    if priority in {"performance", "engine", "engine and performance"}:
        if _number(power1) is not None and _number(power2) is not None:
            delta = abs(_number(power1) - _number(power2))
            stronger = n1 if _number(power1) > _number(power2) else n2 if _number(power2) > _number(power1) else None
            first = (f"{stronger} has the higher listed output ({power1} vs {power2} BHP)." if stronger else f"Both list the same power output ({power1} BHP).")
        else:
            first = f"Listed power is {power1} for {n1} and {power2} for {n2}; the dataset does not provide enough numeric information to rank performance confidently."
        second = f"Transmission: {trans1} vs {trans2}; drivetrain: {drive1} vs {drive2}."
    elif priority in {"city", "city driving"}:
        if _number(mileage1) is not None and _number(mileage2) is not None:
            better = n1 if _number(mileage1) > _number(mileage2) else n2 if _number(mileage2) > _number(mileage1) else None
            first = (f"{better} has the higher listed city mileage ({mileage1} vs {mileage2})." if better else f"Both list the same city mileage ({mileage1}).")
        else:
            first = f"City mileage is {mileage1} for {n1} and {mileage2} for {n2}; the available data does not establish a clear economy winner."
        second = f"Transmission: {trans1} vs {trans2}; compare dimensions and parking needs as well before choosing for city use."
    elif priority in {"highway", "highway trips"}:
        if _number(highway1) is not None and _number(highway2) is not None:
            better = n1 if _number(highway1) > _number(highway2) else n2 if _number(highway2) > _number(highway1) else None
            first = (f"{better} has the higher listed highway mileage ({highway1} vs {highway2})." if better else f"Both list the same highway mileage ({highway1}).")
        else:
            first = f"Highway mileage is {highway1} for {n1} and {highway2} for {n2}; highway economy cannot be ranked confidently from these records."
        second = f"Power is {power1} vs {power2}; use a test drive to assess high-speed comfort and stability, which these figures alone cannot confirm."
    elif priority in {"offroad", "off-road", "off-road driving"}:
        capable1 = any(x in drive1.casefold() for x in ("4wd", "4x4", "awd", "4-wheel", "four wheel"))
        capable2 = any(x in drive2.casefold() for x in ("4wd", "4x4", "awd", "4-wheel", "four wheel"))
        if capable1 and not capable2:
            first = f"{n1} is listed with {drive1}, while {n2} is listed with {drive2}; that gives {n1} the stronger drivetrain specification for off-road use."
        elif capable2 and not capable1:
            first = f"{n2} is listed with {drive2}, while {n1} is listed with {drive1}; that gives {n2} the stronger drivetrain specification for off-road use."
        elif capable1 and capable2:
            first = f"Both vehicles list AWD/4WD-type drivetrains ({drive1} vs {drive2}); the dataset alone does not establish which has better ground clearance, approach angles or off-road hardware."
        else:
            first = f"The dataset lists {drive1} for {n1} and {drive2} for {n2}; neither record clearly confirms AWD/4WD, so do not assume either is suited to demanding trails."
        second = f"Both are listed as {body1} and {body2} respectively; verify ground clearance, tyres and off-road equipment before driving on difficult terrain."
    elif priority in {"economy", "fuel economy", "fuel economy and running costs", "cost"}:
        pnum1, pnum2 = _number(price1), _number(price2)
        if pnum1 is not None and pnum2 is not None:
            cheaper = n1 if pnum1 < pnum2 else n2 if pnum2 < pnum1 else None
            first = (f"{cheaper} has the lower listed price ({price1} vs {price2})." if cheaper else f"Both list the same price ({price1}).")
        else:
            first = f"Listed prices are {price1} for {n1} and {price2} for {n2}; a price winner cannot be confirmed from the available values."
        second = f"City mileage is {mileage1} vs {mileage2}; fuel types are {fuel1} vs {fuel2}. Actual running costs depend on local fuel prices and driving conditions."
    else:  # family / comfort
        if _number(seats1) is not None and _number(seats2) is not None:
            first = f"Seating is listed as {seats1} for {n1} and {seats2} for {n2}; body styles are {body1} and {body2}."
        else:
            first = f"Body styles are listed as {body1} for {n1} and {body2} for {n2}; seating data is {seats1} vs {seats2}."
        second = f"For family use, compare rear-seat space, child-seat fit, boot capacity and safety equipment in person; these records alone do not prove overall comfort."

    return f"{first} {second}"


def build_slm_comparison_prompt(car1, car2, priority):
    """Format comparison facts like the SLM's training examples."""
    aspect_by_priority = {
        "family": "Family use and comfort",
        "city": "City driving",
        "highway": "Highway trips",
        "offroad": "Off-road driving",
        "economy": "Fuel economy and running costs",
        "performance": "Engine and performance",
    }
    aspect = aspect_by_priority.get(str(priority or "family").casefold(), "Family use and comfort")
    instruction = (
        "You are AutoNex AI, an expert automotive consultant. Compare the two provided vehicles "
        "specifically focusing on the requested aspect and produce a clear, objective, and "
        "insightful summary explaining which car is better suited and why."
    )

    def vehicle_block(car, label):
        specs = car["specifications"]
        make = _spec(specs, "Make")
        model = _spec(specs, "Model")
        variant = _spec(specs, "Trim_Variant", "Variant")
        base_name = " ".join(part for part in (make, model) if part != "Not available")
        display_name = base_name or car["name"]
        if variant != "Not available":
            display_name = f"{display_name} ({variant})"

        price = _spec(specs, "Formatted_Price", "Ex_Showroom_Price_INR", "Price")
        displacement = _spec(specs, "Displacement", "Engine_CC", "Engine_Displacement")
        power = _spec(specs, "Power_BHP", "Power", "BHP", "Horsepower")
        body = _spec(specs, "Body_Style", "Body style", "Body_Type")
        seats = _spec(specs, "Seating_Capacity", "Seating capacity", "Seats")
        transmission = _spec(specs, "Transmission", "Transmission_Type")
        drivetrain = _spec(specs, "Drivetrain", "Drive_Type", "Drive")
        fuel = _spec(specs, "Fuel_Type", "Fuel type")
        city = _spec(specs, "Mileage_City", "City mileage", "City_Mileage")
        highway = _spec(specs, "Mileage_Highway", "Highway mileage", "Highway_Mileage")

        return (
            f"[{label}]\n"
            f"Vehicle: {display_name}\n"
            f"- Price: {price}\n"
            f"- Body & Seats: {body}, {seats} Seater\n"
            f"- Powertrain: {displacement}, {power} BHP, {transmission}, {drivetrain}\n"
            f"- Fuel & Mileage: {fuel} | City: {city} | Highway: {highway}"
        )

    user_prompt = (
        f"Compare the following two vehicles with a focus on: '{aspect}'\n\n"
        f"{vehicle_block(car1, 'First Vehicle')}\n\n"
        f"{vehicle_block(car2, 'Second Vehicle')}\n\n"
        f"Aspect: {aspect}\n\n"
        "Provide an expert AutoNex AI comparison summary highlighting key differences, "
        "trade-offs, and which vehicle is better suited for this purpose."
    )
    return f"{instruction}\n\n{user_prompt}\n\n"


def summary_vehicle_names(car):
    """Return the model name the SLM uses, plus its full display name."""
    specs = car["specifications"]
    make = _spec(specs, "Make")
    model = _spec(specs, "Model")
    base_name = " ".join(part for part in (make, model) if part != "Not available")
    return [name for name in (base_name, car["name"]) if name]


@app.route("/api/compare", methods=["POST"])
def compare_cars():
    """Compare two dataset records and use the SLM only if its output is usable."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error_response("Send a valid JSON request.")

    car1_query = data.get("car1", "")
    car2_query = data.get("car2", "")
    priority = data.get("priority", "family")
    if not isinstance(car1_query, str) or not isinstance(car2_query, str):
        return error_response("Both vehicle names must be text.")

    car1_query, car2_query = car1_query.strip(), car2_query.strip()
    allowed_priorities = {"family", "city", "highway", "offroad", "economy", "performance"}
    if not car1_query or not car2_query:
        return error_response("Enter both vehicle names.")
    if len(car1_query) > 150 or len(car2_query) > 150:
        return error_response("Vehicle names must be 150 characters or fewer.")
    if priority not in allowed_priorities:
        priority = "family"

    try:
        # find_car currently returns a pandas Series; convert it before testing truthiness.
        raw1 = find_car(car1_query)
        raw2 = find_car(car2_query)
        if raw1 is None or raw2 is None:
            missing = []
            if raw1 is None:
                missing.append(car1_query)
            if raw2 is None:
                missing.append(car2_query)
            return jsonify({
                "success": False,
                "error": "Could not find: " + ", ".join(missing),
                "hint": "Try the exact model or variant name in your dataset."
            }), 404

        car1 = comparison_car_facts(raw1)
        car2 = comparison_car_facts(raw2)
        if not car1 or not car2:
            return jsonify({"success": False, "error": "Could not normalize one or both vehicle records."}), 500

        fallback = build_comparison_summary(car1, car2, priority)
        analysis = fallback
        model_used = False

        # This character-level model was trained on instruction + vehicle facts,
        # so mirror that format and allow enough characters for its two-sentence answers.
        if SLM_READY:
            try:
                prompt = build_slm_comparison_prompt(car1, car2, priority)
                generated = generate_text(prompt, max_new_tokens=400, temperature=0.15)
                candidate = re.sub(r"\s+", " ", str(generated or "")).strip()
                ends = list(re.finditer(r"[.!?](?=\s|$)", candidate))
                if len(ends) >= 2:
                    # Training answers contain two sentences; discard any drift after them.
                    candidate = candidate[:ends[1].end()].strip()
                lower_candidate = candidate.casefold()
                names_present = all(
                    any(name.casefold() in lower_candidate for name in summary_vehicle_names(car))
                    for car in (car1, car2)
                )
                has_two_sentences = len(re.findall(r"[.!?](?:\s|$)", candidate)) >= 2
                has_enough_words = len(re.findall(r"\b\w+\b", candidate)) >= 24
                looks_like_prompt = any(token in candidate.casefold() for token in ("car 1:", "car 2:", "write a concise", "json"))
                if 120 <= len(candidate) <= 650 and names_present and has_two_sentences and has_enough_words and not looks_like_prompt:
                    analysis = candidate
                    model_used = True
            except Exception:
                logger.exception("SLM comparison generation failed; using grounded fallback.")

        return jsonify({
            "success": True,
            "car1": car1,
            "car2": car2,
            "priority": priority,
            "analysis": analysis,
            "model_used": model_used,
            "model": "AutoNex Tiny GPT" if model_used else "Grounded dataset summary",
            "notice": "Comparison uses available dataset fields. Verify vehicle specifications before purchase."
        }), 200

    except Exception:
        logger.exception("Car comparison failed")
        return jsonify({
            "success": False,
            "error": "Comparison failed. Check the dataset and Flask terminal."
        }), 500


# --------------------------------------------------
# Trained SLM chat endpoint
# --------------------------------------------------

@app.post("/api/chat")
def chat():
    if not SLM_READY:
        return error_response(
            "The SLM is not loaded. Check the checkpoint and backend logs.",
            503,
        )

    data = request.get_json(silent=True) or {}
    prompt = str(
        data.get("message", data.get("prompt", ""))
    ).strip()

    if not prompt:
        return error_response("Provide a message or prompt.")

    if len(prompt) > 2000:
        return error_response("Message is too long.")

    try:
        answer = generate_text(
            prompt,
            max_new_tokens=120,
        )

        return jsonify({
            "success": True,
            "prompt": prompt,
            "response": str(answer).strip(),
        }), 200

    except Exception:
        logger.exception("SLM generation failed.")
        return error_response(
            "Text generation failed. Check the backend terminal.",
            500,
        )


# --------------------------------------------------
# Run local development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
