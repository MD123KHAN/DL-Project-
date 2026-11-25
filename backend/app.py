
"""
AutoNex-AI Flask API
Connects the vehicle dataset, trained TinyGPT SLM,
Wikimedia Commons image search, and frontend.
"""

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from io import BytesIO
import json
import logging
import os
import re
from threading import Lock

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from model import find_car, get_car_suggestions, load_dataset
from slm import load_slm, generate_text


# --------------------------------------------------
# Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoNex-AI")

app.config["JSON_SORT_KEYS"] = False
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES * 3 + 256 * 1024

# SigLIP is loaded only when a user requests image recognition. Render's
# default instance stays protected from the model's large download and memory
# use; a larger deployment can enable it explicitly with an environment var.
_vision_default = "false" if os.getenv("RENDER", "").casefold() == "true" else "true"
VISION_ENABLED = os.getenv("AUTONEX_VISION_ENABLED", _vision_default).casefold() in {
    "1", "true", "yes", "on"
}
VISION_MODEL_ID = os.getenv(
    "AUTONEX_VISION_MODEL",
    "google/siglip-base-patch16-224",
)
_VISION_CLASSIFIER = None
_VISION_ERROR = None
_VISION_LOCK = Lock()
_VEHICLE_IMAGE_LABELS = None
_VEHICLE_IMAGE_LABELS_LOCK = Lock()

# Set to your actual frontend origin if you want to restrict CORS.
# Example: http://127.0.0.1:5500
# Leave unrestricted during local development.
SLM_READY = False
SLM_ERROR = None

# Local car images for fast, reliable demos
IMAGE_DIR = BASE_DIR / "image_"

@app.get("/api/local-car-image/<path:filename>")
def get_local_car_image(filename):
    # Only serve image files from backend/image_/.
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


@app.errorhandler(HTTPException)
def handle_http_exception(error):
    """Keep API routing and method errors in the same JSON format as other API responses."""
    if request.path == "/api" or request.path.startswith("/api/"):
        status_code = error.code or 500
        if status_code == 404:
            message = (
                "API endpoint not found. Restart the backend from this project folder "
                "to load the current routes."
            )
        elif status_code == 405:
            message = "This API endpoint does not support that request method."
        else:
            message = error.description or "The API request could not be completed."
        return error_response(message, status_code)

    return error


@app.errorhandler(Exception)
def handle_unexpected_exception(error):
    """Return JSON for unexpected API failures while preserving normal page errors."""
    if request.path == "/api" or request.path.startswith("/api/"):
        logger.exception("Unhandled API error at %s", request.path)
        return error_response(
            "The API could not complete this request. Check the Flask terminal for details.",
            500,
        )

    raise error


@app.errorhandler(413)
def request_too_large(_error):
    return error_response(
        "The uploaded file is too large. Choose an image smaller than 8 MB.",
        413,
    )


def load_vision_classifier():
    """Load the local zero-shot image model once, on the first upload."""
    global _VISION_CLASSIFIER, _VISION_ERROR

    if _VISION_CLASSIFIER is not None:
        return _VISION_CLASSIFIER

    with _VISION_LOCK:
        if _VISION_CLASSIFIER is not None:
            return _VISION_CLASSIFIER

        try:
            from transformers import pipeline

            _VISION_CLASSIFIER = pipeline(
                "zero-shot-image-classification",
                model=VISION_MODEL_ID,
                device=-1,
            )
            _VISION_ERROR = None
            return _VISION_CLASSIFIER
        except Exception as exc:
            _VISION_ERROR = str(exc)
            logger.exception("Could not load the local vehicle vision model.")
            cause = next(
                (line.strip() for line in str(exc).splitlines() if line.strip()),
                type(exc).__name__,
            )
            raise RuntimeError(
                f"The local image-recognition model could not be loaded ({cause}). "
                "Start Flask with the project interpreter: "
                ".venv\\Scripts\\python.exe backend\\app.py. "
                "If a dependency is missing, install requirements into that same .venv."
            ) from exc


def vehicle_image_labels():
    """Build image labels from dataset make/model pairs, not guessed trims."""
    global _VEHICLE_IMAGE_LABELS
    if _VEHICLE_IMAGE_LABELS is not None:
        return list(_VEHICLE_IMAGE_LABELS)

    with _VEHICLE_IMAGE_LABELS_LOCK:
        if _VEHICLE_IMAGE_LABELS is None:
            df = load_dataset()
            labels = {}
            for make, model_name in df[["Make", "Model"]].itertuples(index=False, name=None):
                make = str(make or "").strip()
                model_name = str(model_name or "").strip()
                if not make or not model_name or make.casefold() == "nan" or model_name.casefold() == "nan":
                    continue
                name = f"{make} {model_name}"
                labels.setdefault(name.casefold(), name)
            _VEHICLE_IMAGE_LABELS = tuple(sorted(labels.values(), key=str.casefold))

    return list(_VEHICLE_IMAGE_LABELS)


def vehicle_candidate_map():
    """Create the shared vehicle/category labels used by image matching."""
    vehicle_names = vehicle_image_labels()
    if not vehicle_names:
        return {}
    candidates = {
        f"{name} car": {"type": "vehicle", "name": name}
        for name in vehicle_names
    }
    for category in (
        "SUV", "sedan", "hatchback", "pickup truck", "coupe",
        "convertible", "van", "wagon", "minivan", "sports car",
    ):
        candidates[f"{category} vehicle"] = {
            "type": "category",
            "name": category,
        }
    return candidates


def prepare_vision_image(image):
    """Downscale before inference; SigLIP itself resizes inputs to 224 px."""
    image = ImageOps.exif_transpose(image)
    image.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    return image.convert("RGB")


def format_vision_predictions(ranked, candidate_map):
    """Convert pipeline scores to the same response shape for each image."""
    predictions = []
    for item in sorted(
        ranked,
        key=lambda result: float(result.get("score", 0.0)),
        reverse=True,
    )[:5]:
        candidate = candidate_map.get(item.get("label"))
        if candidate:
            predictions.append({
                **candidate,
                "score": round(float(item.get("score", 0.0)), 4),
            })

    vehicle_predictions = [item for item in predictions if item["type"] == "vehicle"]
    best_vehicle = vehicle_predictions[0] if vehicle_predictions else None
    top_overall = predictions[0] if predictions else None
    uncertain = (
        best_vehicle is None
        or best_vehicle["score"] < 0.18
        or top_overall is None
        or top_overall["type"] != "vehicle"
    )
    return {
        "suggested_vehicle": best_vehicle["name"] if best_vehicle else "",
        "confidence": best_vehicle["score"] if best_vehicle else 0.0,
        "uncertain": uncertain,
        "predictions": predictions,
    }


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
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api")
def api_root():
    return jsonify({
        "application": "AutoNex-AI",
        "status": "running",
        "slm_loaded": SLM_READY,
        "endpoints": {
            "health": "/api/health",
            "car_search": "/api/car?q=Mahindra%20Thar",
            "suggestions": "/api/suggestions?q=Mahindra%20Thar",
            "vision_warmup": "/api/vision/warmup",
            "image_recognition": "/api/recognize",
            "image_recognition_batch": "/api/recognize/batch",
            "car_image": "/api/car-image?q=Mahindra%20Thar",
            "chat": "/api/chat",
        },
        "vision_enabled": VISION_ENABLED,
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
        "vision_enabled": VISION_ENABLED,
        "vision_model_loaded": _VISION_CLASSIFIER is not None,
    }), 200 if dataset_ready else 503


@app.get("/api/suggestions")
def get_suggestions():
    """Return dataset-backed make/model/variant/year autocomplete options."""
    query = request.args.get("q", "").strip()
    if len(query) > 150:
        return error_response("Search query is too long.")

    try:
        limit = int(request.args.get("limit", "10"))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 10))

    try:
        suggestions = get_car_suggestions(query, limit=limit)
        return jsonify({"success": True, "suggestions": suggestions}), 200
    except Exception:
        logger.exception("Vehicle suggestions failed.")
        return error_response("Vehicle suggestions are unavailable.", 500)


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

        # Preserve an explicit year so model.py can filter candidates before
        # choosing a matching variant. Only retry without it if necessary.
        car = find_car(raw_query)

        if car is None:
            car = find_car(query)

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


@app.post("/api/vision/warmup")
def warmup_vehicle_vision():
    """Load the vision model before the user asks it to classify a photo."""
    if not VISION_ENABLED:
        return error_response(
            "Image recognition is disabled on this server.",
            503,
        )
    try:
        if not vehicle_candidate_map():
            return error_response("No vehicle labels are available in the dataset.", 503)
        load_vision_classifier()
        return jsonify({
            "success": True,
            "ready": True,
            "model": VISION_MODEL_ID,
        }), 200
    except RuntimeError as exc:
        return error_response(str(exc), 503)


@app.post("/api/recognize")
def recognize_vehicle_image():
    """Suggest dataset make/model labels from an image; never return specs."""
    if not VISION_ENABLED:
        return error_response(
            "Image recognition is disabled on this server. Search for the vehicle by name instead.",
            503,
        )

    upload = request.files.get("image")
    if upload is None:
        return error_response("Choose a vehicle image to continue.")
    if not upload.filename:
        return error_response("Choose a vehicle image to continue.")
    if upload.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        return error_response("Use a JPEG, PNG, or WebP image.", 415)

    image_bytes = upload.stream.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return error_response("Choose an image smaller than 8 MB.", 413)
    if not image_bytes:
        return error_response("The selected image is empty.")

    try:
        image = Image.open(BytesIO(image_bytes))
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            return error_response("Use a JPEG, PNG, or WebP image.", 415)
        if image.width * image.height > MAX_IMAGE_PIXELS:
            return error_response("Choose an image with 20 megapixels or fewer.", 413)
        image.load()
        image = prepare_vision_image(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return error_response("The selected file is not a readable image.")

    try:
        candidate_map = vehicle_candidate_map()
        if not candidate_map:
            return error_response("No vehicle labels are available in the dataset.", 503)

        classifier = load_vision_classifier()
        ranked = classifier(image, candidate_labels=list(candidate_map))
        result = format_vision_predictions(ranked, candidate_map)

        return jsonify({
            "success": True,
            "model": VISION_MODEL_ID,
            **result,
            "notice": (
                "These are relative image-match scores, not verified identity. "
                "Confirm or correct the vehicle before loading dataset specifications. "
                "A photo cannot reveal year, trim, price, mileage, engine power, or other specifications."
            ),
        }), 200
    except RuntimeError as exc:
        return error_response(str(exc), 503)
    except Exception:
        logger.exception("Vehicle image recognition failed.")
        return error_response(
            "Image recognition failed. You can still search for the vehicle by name.",
            500,
        )


@app.post("/api/recognize/batch")
def recognize_vehicle_images_batch():
    """Recognize exactly three uploaded images in one model batch."""
    if not VISION_ENABLED:
        return error_response(
            "Image recognition is disabled on this server.",
            503,
        )

    uploads = request.files.getlist("images")
    if len(uploads) != 3:
        return error_response("Upload exactly three car photos to compare them together.")

    images = []
    for upload in uploads:
        if not upload or not upload.filename:
            return error_response("Choose a photo for each of the three cars.")
        if upload.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
            return error_response("Use JPEG, PNG, or WebP images for all three cars.", 415)

        image_bytes = upload.stream.read(MAX_IMAGE_BYTES + 1)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            return error_response("Each image must be smaller than 8 MB.", 413)
        if not image_bytes:
            return error_response("One of the selected images is empty.")

        try:
            image = Image.open(BytesIO(image_bytes))
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                return error_response("Use JPEG, PNG, or WebP images for all three cars.", 415)
            if image.width * image.height > MAX_IMAGE_PIXELS:
                return error_response("Each image must have 20 megapixels or fewer.", 413)
            image.load()
            images.append(prepare_vision_image(image))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            return error_response("One of the selected files is not a readable image.")

    try:
        candidate_map = vehicle_candidate_map()
        if not candidate_map:
            return error_response("No vehicle labels are available in the dataset.", 503)

        classifier = load_vision_classifier()
        ranked_batch = classifier(
            images,
            candidate_labels=list(candidate_map),
            batch_size=3,
        )
        if len(ranked_batch) != 3:
            raise RuntimeError("The vision model did not return one result per photo.")

        return jsonify({
            "success": True,
            "model": VISION_MODEL_ID,
            "results": [
                format_vision_predictions(ranked, candidate_map)
                for ranked in ranked_batch
            ],
            "notice": (
                "Scores are relative image matches, not verified identity. "
                "Confirm each vehicle and dataset variant before comparing specifications."
            ),
        }), 200
    except RuntimeError as exc:
        return error_response(str(exc), 503)
    except Exception:
        logger.exception("Three-photo vehicle recognition failed.")
        return error_response(
            "Three-photo recognition failed. Check the Flask terminal.",
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


def _measurement_unit(value):
    """Return the unit text after a leading numeric measurement, if present."""
    match = re.match(r"^\s*-?\d[\d,]*(?:\.\d+)?\s*(.*?)\s*$", str(value or ""))
    return re.sub(r"[^a-z%]+", "", match.group(1).casefold()) if match else ""


def _compare_numeric(first, second, *, higher_is_better=True, compare_units=False):
    """Compare two displayed numeric values, returning first/second/tie/unknown."""
    value1, value2 = _number(first), _number(second)
    if value1 is None or value2 is None:
        return "unknown"
    if compare_units and _measurement_unit(first) != _measurement_unit(second):
        return "incomparable"
    if value1 == value2:
        return "tie"
    first_better = value1 > value2 if higher_is_better else value1 < value2
    return "first" if first_better else "second"


def build_five_basis_comparison(car1, car2, priority):
    """Build five explicit, deterministic comparisons from the selected records."""
    specs1, specs2 = car1["specifications"], car2["specifications"]
    name1, name2 = car1["name"], car2["name"]

    def unavailable(value):
        return value == "Not available"

    def make_basis(label, result, summary, value1, value2):
        return {
            "label": label,
            "result": result,
            "summary": summary,
            "car1_value": value1,
            "car2_value": value2,
        }

    # 1. Affordability: compare canonical INR values, not rounded lakh strings.
    price1 = _spec(specs1, "Formatted_Price", "Ex_Showroom_Price_INR", "Price")
    price2 = _spec(specs2, "Formatted_Price", "Ex_Showroom_Price_INR", "Price")
    price_inr1 = _spec(specs1, "Ex_Showroom_Price_INR", "Price_INR")
    price_inr2 = _spec(specs2, "Ex_Showroom_Price_INR", "Price_INR")
    price_result = _compare_numeric(price_inr1, price_inr2, higher_is_better=False)
    if price_result == "first":
        price_verdict = "car1"
        price_summary = f"{name1} has the lower listed ex-showroom price. This does not include insurance, taxes, or ownership costs."
    elif price_result == "second":
        price_verdict = "car2"
        price_summary = f"{name2} has the lower listed ex-showroom price. This does not include insurance, taxes, or ownership costs."
    elif price_result == "tie":
        price_verdict = "tie"
        price_summary = "Both selected records list the same ex-showroom price."
    else:
        price_verdict = "incomplete"
        price_summary = "A price advantage cannot be determined because a comparable numeric price is missing."
    price_basis = make_basis("Price & affordability", price_verdict, price_summary, price1, price2)

    # 2. Economy: compare city/highway mileage only when fuels and units match.
    city1 = _spec(specs1, "Mileage_City", "City mileage", "City_Mileage")
    city2 = _spec(specs2, "Mileage_City", "City mileage", "City_Mileage")
    highway1 = _spec(specs1, "Mileage_Highway", "Highway mileage", "Highway_Mileage")
    highway2 = _spec(specs2, "Mileage_Highway", "Highway mileage", "Highway_Mileage")
    fuel1 = _spec(specs1, "Fuel_Type", "Fuel type")
    fuel2 = _spec(specs2, "Fuel_Type", "Fuel type")
    same_fuel = not unavailable(fuel1) and not unavailable(fuel2) and fuel1.casefold() == fuel2.casefold()
    priority_key = str(priority or "family").casefold()
    if priority_key in {"city", "city driving"}:
        economy_metrics = [("City", city1, city2)]
    elif priority_key in {"highway", "highway trips"}:
        economy_metrics = [("Highway", highway1, highway2)]
    else:
        economy_metrics = [("City", city1, city2), ("Highway", highway1, highway2)]

    economy_outcomes = []
    economy_notes = []
    for metric_name, metric1, metric2 in economy_metrics:
        if unavailable(metric1) or unavailable(metric2):
            economy_outcomes.append("unknown")
            economy_notes.append(f"{metric_name.lower()} mileage is missing for at least one vehicle")
            continue
        if not same_fuel:
            economy_outcomes.append("incomparable")
            economy_notes.append(f"{metric_name.lower()} values are shown, but fuel types differ or are unavailable")
            continue
        outcome = _compare_numeric(metric1, metric2, compare_units=True)
        economy_outcomes.append(outcome)
        if outcome == "first":
            economy_notes.append(f"{name1} has the higher listed {metric_name.lower()} mileage ({metric1} vs {metric2})")
        elif outcome == "second":
            economy_notes.append(f"{name2} has the higher listed {metric_name.lower()} mileage ({metric2} vs {metric1})")
        elif outcome == "tie":
            economy_notes.append(f"both list the same {metric_name.lower()} mileage ({metric1})")
        else:
            economy_notes.append(f"{metric_name.lower()} mileage units do not match, so the figures are not directly comparable")

    comparable_economy = [item for item in economy_outcomes if item in {"first", "second", "tie"}]
    directions = set(comparable_economy) - {"tie"}
    if not comparable_economy:
        economy_verdict = "incomparable" if "incomparable" in economy_outcomes else "incomplete"
    elif not directions:
        economy_verdict = "tie" if len(comparable_economy) == len(economy_outcomes) else "partial_tie"
    elif directions == {"first"}:
        economy_verdict = "car1" if len(comparable_economy) == len(economy_outcomes) else "partial_car1"
    elif directions == {"second"}:
        economy_verdict = "car2" if len(comparable_economy) == len(economy_outcomes) else "partial_car2"
    else:
        economy_verdict = "mixed"
    economy_summary = "; ".join(economy_notes)
    if economy_summary:
        economy_summary = economy_summary[0].upper() + economy_summary[1:] + "."
    economy_basis = make_basis(
        "Fuel economy",
        economy_verdict,
        economy_summary,
        f"City: {city1} · Highway: {highway1} · Fuel: {fuel1}",
        f"City: {city2} · Highway: {highway2} · Fuel: {fuel2}",
    )

    # 3. Performance: power is the ranking field; other engine fields are context.
    power1 = _spec(specs1, "Power_BHP", "Power", "BHP", "Horsepower")
    power2 = _spec(specs2, "Power_BHP", "Power", "BHP", "Horsepower")
    displacement1 = _spec(specs1, "Displacement", "Engine_CC", "Engine_Displacement")
    displacement2 = _spec(specs2, "Displacement", "Engine_CC", "Engine_Displacement")
    cylinders1 = _spec(specs1, "Cylinders", "Cylinder_Count")
    cylinders2 = _spec(specs2, "Cylinders", "Cylinder_Count")
    performance_result = _compare_numeric(power1, power2)
    if performance_result == "first":
        performance_verdict = "car1"
        performance_summary = f"{name1} has higher listed power ({power1} BHP vs {power2} BHP). Power alone does not establish acceleration or handling."
    elif performance_result == "second":
        performance_verdict = "car2"
        performance_summary = f"{name2} has higher listed power ({power2} BHP vs {power1} BHP). Power alone does not establish acceleration or handling."
    elif performance_result == "tie":
        performance_verdict = "tie"
        performance_summary = f"Both records list {power1} BHP. Engine displacement and cylinders are shown for context, not used to declare an overall winner."
    else:
        performance_verdict = "incomplete"
        performance_summary = "A power comparison is unavailable because one or both records lack a numeric power value."
    performance_basis = make_basis(
        "Power & engine",
        performance_verdict,
        performance_summary,
        f"Power: {power1} BHP · Engine: {displacement1} · Cylinders: {cylinders1}",
        f"Power: {power2} BHP · Engine: {displacement2} · Cylinders: {cylinders2}",
    )

    # 4. Practicality: seat count is descriptive; more seats are not a comfort score.
    seats1 = _spec(specs1, "Seating_Capacity", "Seating capacity", "Seats")
    seats2 = _spec(specs2, "Seating_Capacity", "Seating capacity", "Seats")
    body1 = _spec(specs1, "Body_Style", "Body style", "Body_Type")
    body2 = _spec(specs2, "Body_Style", "Body style", "Body_Type")
    doors1 = _spec(specs1, "Doors", "Door_Count")
    doors2 = _spec(specs2, "Doors", "Door_Count")
    seat_result = _compare_numeric(seats1, seats2)
    if seat_result == "first":
        practicality_verdict = "car1"
        practicality_summary = f"{name1} has more listed seats ({seats1} vs {seats2}); this measures capacity, not passenger comfort."
    elif seat_result == "second":
        practicality_verdict = "car2"
        practicality_summary = f"{name2} has more listed seats ({seats2} vs {seats1}); this measures capacity, not passenger comfort."
    elif seat_result == "tie":
        practicality_verdict = "tie"
        practicality_summary = f"Both list {seats1} seats. Body style and door count are shown for layout context; cabin dimensions and boot space are not in these records."
    else:
        practicality_verdict = "incomplete"
        practicality_summary = "Seating capacity is missing, so this category cannot be ranked. Body style and door count remain available as descriptive details."
    practicality_basis = make_basis(
        "Space & practicality",
        practicality_verdict,
        practicality_summary,
        f"Seats: {seats1} · Body: {body1} · Doors: {doors1}",
        f"Seats: {seats2} · Body: {body2} · Doors: {doors2}",
    )

    # 5. Capability and controls: only make a directional claim for off-road priority.
    drivetrain1 = _spec(specs1, "Drivetrain", "Drive_Type", "Drive")
    drivetrain2 = _spec(specs2, "Drivetrain", "Drive_Type", "Drive")
    transmission1 = _spec(specs1, "Transmission", "Transmission_Type")
    transmission2 = _spec(specs2, "Transmission", "Transmission_Type")
    has_4wd1 = any(token in drivetrain1.casefold() for token in ("4wd", "4x4", "awd", "4-wheel", "four wheel"))
    has_4wd2 = any(token in drivetrain2.casefold() for token in ("4wd", "4x4", "awd", "4-wheel", "four wheel"))
    is_offroad = priority_key in {"offroad", "off-road", "off-road driving"}
    if is_offroad and has_4wd1 != has_4wd2:
        capability_verdict = "car1" if has_4wd1 else "car2"
        better_name = name1 if has_4wd1 else name2
        capability_summary = f"{better_name} has the listed AWD/4WD drivetrain advantage for off-road use. Ground clearance, tyres, low-range gearing, and off-road equipment are not established by these fields."
    elif is_offroad and has_4wd1 and has_4wd2:
        capability_verdict = "shared"
        capability_summary = "Both records list AWD/4WD. These records do not distinguish ground clearance, tyres, low-range gearing, or other off-road equipment."
    elif is_offroad:
        capability_verdict = "incomplete"
        capability_summary = "Neither record clearly lists AWD/4WD; off-road suitability cannot be determined from drivetrain alone."
    elif drivetrain1 == drivetrain2 and transmission1 == transmission2:
        capability_verdict = "tie"
        capability_summary = "Both records list the same drivetrain and transmission. This category does not distinguish the vehicles for the selected priority."
    else:
        capability_verdict = "preference"
        capability_summary = "The listed drivetrain and transmission differ; which setup suits you depends on your driving conditions and preference."
    capability_basis = make_basis(
        "Drivetrain & transmission",
        capability_verdict,
        capability_summary,
        f"{drivetrain1} · {transmission1}",
        f"{drivetrain2} · {transmission2}",
    )

    return [price_basis, economy_basis, performance_basis, practicality_basis, capability_basis]


def build_priority_comparison(car1, car2, basis, priority):
    """Return one clear, data-backed result for the user's selected priority."""
    specs1, specs2 = car1["specifications"], car2["specifications"]
    name1, name2 = car1["name"], car2["name"]
    priority_key = str(priority or "family").casefold()
    priority_labels = {
        "family": "Family use and comfort",
        "city": "City driving",
        "highway": "Highway trips",
        "offroad": "Off-road driving",
        "economy": "Fuel economy and running costs",
        "performance": "Engine and performance",
    }
    focus_by_priority = {
        "family": "Space & practicality",
        "city": "Fuel economy",
        "highway": "Fuel economy",
        "offroad": "Drivetrain & transmission",
        "economy": "Fuel economy",
        "performance": "Power & engine",
    }
    priority_label = priority_labels.get(priority_key, "your selected priority")
    focus_label = focus_by_priority.get(priority_key, "Space & practicality")
    selected = next(item for item in basis if item["label"] == focus_label).copy()

    def verdict_for(result):
        if result == "car1":
            return name1
        if result == "car2":
            return name2
        return None

    if priority_key in {"city", "highway"}:
        metric = "City" if priority_key == "city" else "Highway"
        field = "Mileage_City" if priority_key == "city" else "Mileage_Highway"
        value1 = _spec(specs1, field, f"{metric} mileage", f"{metric}_Mileage")
        value2 = _spec(specs2, field, f"{metric} mileage", f"{metric}_Mileage")
        fuel1 = _spec(specs1, "Fuel_Type", "Fuel type")
        fuel2 = _spec(specs2, "Fuel_Type", "Fuel type")
        same_fuel = fuel1 != "Not available" and fuel2 != "Not available" and fuel1.casefold() == fuel2.casefold()
        selected["label"] = f"{metric} fuel economy"
        selected["car1_value"], selected["car2_value"] = value1, value2
        if not same_fuel:
            selected["result"] = "incomparable"
            selected["summary"] = f"{metric} mileage cannot fairly decide this priority because the fuel types differ or are missing ({fuel1} vs {fuel2})."
        else:
            outcome = _compare_numeric(value1, value2, compare_units=True)
            selected["result"] = {"first": "car1", "second": "car2", "tie": "tie"}.get(outcome, "incomplete")
            if outcome == "first":
                selected["summary"] = f"{name1} is the better match for {priority_label.lower()} on the listed {metric.lower()} mileage ({value1} vs {value2}); both records list {fuel1}."
            elif outcome == "second":
                selected["summary"] = f"{name2} is the better match for {priority_label.lower()} on the listed {metric.lower()} mileage ({value2} vs {value1}); both records list {fuel1}."
            elif outcome == "tie":
                selected["summary"] = f"Neither car wins for {priority_label.lower()}: both list {value1} {metric.lower()} mileage and the same fuel type ({fuel1})."
            elif outcome == "incomparable":
                selected["summary"] = f"No winner for {priority_label.lower()}: the mileage units differ ({value1} vs {value2}), so the figures are not directly comparable."
            else:
                selected["summary"] = f"No winner for {priority_label.lower()}: a comparable numeric {metric.lower()} mileage value is missing."

    elif priority_key == "economy":
        price = next(item for item in basis if item["label"] == "Price & affordability")
        mileage = next(item for item in basis if item["label"] == "Fuel economy")
        price_result, mileage_result = price["result"], mileage["result"]
        decisive = {"car1", "car2", "tie"}
        if mileage_result == "mixed":
            result = "mixed"
        elif price_result in decisive and mileage_result in decisive:
            edges = {result for result in (price_result, mileage_result) if result != "tie"}
            result = next(iter(edges)) if len(edges) == 1 else ("tie" if not edges else "mixed")
        else:
            result = "incomplete"
        selected.update({
            "label": "Fuel economy and running costs",
            "result": result,
            "car1_value": f"Price: {price['car1_value']} · {mileage['car1_value']}",
            "car2_value": f"Price: {price['car2_value']} · {mileage['car2_value']}",
        })
        if result in {"car1", "car2"}:
            winner = name1 if result == "car1" else name2
            winning_edges = []
            if price_result == result:
                winning_edges.append(price["summary"])
            if mileage_result == result:
                winning_edges.append(mileage["summary"])
            selected["summary"] = f"{winner} is the stronger match for fuel economy and running costs. " + " ".join(winning_edges) + " These figures compare listed price and mileage; actual running cost also depends on fuel price, annual distance, and local taxes."
        elif result == "tie":
            selected["summary"] = f"No winner for fuel economy and running costs: both records tie on listed ex-showroom price and comparable mileage. Actual ownership cost can still differ with fuel price, annual distance, insurance, and taxes."
        elif result == "mixed":
            selected["summary"] = f"There is no single winner for fuel economy and running costs because the listed figures point in different directions. {price['summary']} {mileage['summary']} Actual total cost depends on expected annual distance and fuel prices."
        else:
            selected["summary"] = f"No reliable winner for fuel economy and running costs: price or comparable mileage data is missing or uses incompatible fuel types. {price['summary']} {mileage['summary']}"

    elif priority_key == "family":
        seats1 = _spec(specs1, "Seating_Capacity", "Seating capacity", "Seats")
        seats2 = _spec(specs2, "Seating_Capacity", "Seating capacity", "Seats")
        selected["car1_value"], selected["car2_value"] = f"{seats1} seats", f"{seats2} seats"
        result = selected["result"]
        if result in {"car1", "car2"}:
            winner = verdict_for(result)
            winner_seats = seats1 if result == "car1" else seats2
            other_seats = seats2 if result == "car1" else seats1
            selected["summary"] = f"{winner} is the better match for family capacity, with {winner_seats} listed seats versus {other_seats}. Seat count measures capacity only; comfort, child-seat fit, and boot space are not available in these records."
        elif result == "tie":
            selected["summary"] = f"Neither car wins on listed family capacity: both have {seats1} seats. The records do not establish comfort, child-seat fit, or boot space."

    elif priority_key == "performance":
        power1 = _spec(specs1, "Power_BHP", "Power", "BHP", "Horsepower")
        power2 = _spec(specs2, "Power_BHP", "Power", "BHP", "Horsepower")
        selected["car1_value"], selected["car2_value"] = f"{power1} BHP", f"{power2} BHP"
        result = selected["result"]
        if result in {"car1", "car2"}:
            winner = verdict_for(result)
            winner_power = power1 if result == "car1" else power2
            other_power = power2 if result == "car1" else power1
            selected["summary"] = f"{winner} leads on the listed power figure ({winner_power} BHP vs {other_power} BHP), so it is the specification winner for engine and performance. Power alone does not establish acceleration, handling, or real-world performance."

    result = selected["result"]
    winner = verdict_for(result)
    if winner:
        if priority_key == "family":
            winner_value = selected["car1_value"] if result == "car1" else selected["car2_value"]
            other_value = selected["car2_value"] if result == "car1" else selected["car1_value"]
            headline = f"More listed seats: {winner} ({winner_value} vs {other_value})"
        elif priority_key in {"city", "highway"}:
            winner_value = selected["car1_value"] if result == "car1" else selected["car2_value"]
            other_value = selected["car2_value"] if result == "car1" else selected["car1_value"]
            headline = f"Mileage winner for {priority_label.lower()}: {winner} ({winner_value} vs {other_value})"
        elif priority_key == "performance":
            winner_value = selected["car1_value"] if result == "car1" else selected["car2_value"]
            other_value = selected["car2_value"] if result == "car1" else selected["car1_value"]
            headline = f"Performance specification winner: {winner} ({winner_value} vs {other_value})"
        elif priority_key == "offroad":
            headline = f"Stronger listed off-road drivetrain: {winner}"
        elif priority_key == "economy":
            headline = f"Stronger listed value for {priority_label.lower()}: {winner}"
        else:
            headline = f"Winner for {priority_label}: {winner}"
    elif result == "tie":
        headline = f"No winner for {priority_label}: the listed figures are tied"
    elif result == "shared":
        headline = f"No winner for {priority_label}: both records list the same drivetrain capability"
    elif result == "mixed":
        headline = f"No single winner for {priority_label}: the results are split"
    elif result == "preference":
        headline = f"No single winner for {priority_label}: the listed setups suit different preferences"
    elif result == "incomparable":
        headline = f"No winner for {priority_label}: the listed figures are not directly comparable"
    else:
        headline = f"No clear winner for {priority_label}: the available data is insufficient"

    selected["priority"] = priority_key
    selected["priority_label"] = priority_label
    selected["winner"] = winner
    selected["headline"] = headline
    return selected


@app.route("/api/compare", methods=["POST"])
def compare_cars():
    """Compare two selected records on the user's chosen priority."""
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

        all_basis = build_five_basis_comparison(car1, car2, priority)
        priority_result = build_priority_comparison(car1, car2, all_basis, priority)
        analysis = priority_result["summary"]

        return jsonify({
            "success": True,
            "car1": car1,
            "car2": car2,
            "priority": priority,
            "priority_result": priority_result,
            "analysis": analysis,
            # Keep the existing field for API compatibility, but return only the
            # selected priority so clients cannot accidentally show unrelated cards.
            "comparison_basis": [priority_result],
            "model_used": False,
            "model": "Priority-focused rule-based comparison",
            "notice": "Facts come from the selected vehicle records. Missing fields are not estimated; family comfort, total ownership cost, and real-world performance may require additional information."
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
