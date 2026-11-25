
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);
const IS_LOCAL_FRONTEND =
    window.location.protocol === "file:" ||
    (LOCAL_HOSTS.has(window.location.hostname.toLowerCase()) &&
        window.location.port !== "5000");
const API_BASE = IS_LOCAL_FRONTEND
    ? "http://127.0.0.1:5000"
    : window.location.origin;
const API_URL = `${API_BASE}/api/car`;
const SUGGESTIONS_URL = `${API_BASE}/api/suggestions`;
const IMAGE_URL = `${API_BASE}/api/car-image`;

// DOM elements
const searchForm = document.getElementById("searchForm");
const carInput = document.getElementById("carInput");
const searchButton = document.getElementById("searchButton");
const statusBox = document.getElementById("status");
const results = document.getElementById("results");

const suggestionList = document.getElementById("suggestionList");
const carName = document.getElementById("carName");
const manufacturerLabel = document.getElementById("manufacturerLabel");
const carDescription = document.getElementById("carDescription");
const carImage = document.getElementById("carImage");
const imagePlaceholder = document.getElementById("imagePlaceholder");
const imageAttribution = document.getElementById("imageAttribution");
const specGrid = document.getElementById("specGrid");
const sourceLink = document.getElementById("sourceLink");
const sourceText = document.getElementById("sourceText");
const vehiclePhotoInput = document.getElementById("vehiclePhotoInput");
const vehiclePhotoPreview = document.getElementById("vehiclePhotoPreview");
const uploadPreviewPanel = document.getElementById("uploadPreviewPanel");
const recognizeVehicleButton = document.getElementById("recognizeVehicleButton");
const recognitionStatus = document.getElementById("recognitionStatus");
const recognitionResult = document.getElementById("recognitionResult");
const recognitionVehicleSelect = document.getElementById("recognitionVehicleSelect");
const loadRecognizedVehicleButton = document.getElementById("loadRecognizedVehicleButton");

let suggestionItems = [];
let activeSuggestionIndex = -1;
let debounceTimer = null;
let latestSuggestionRequest = 0;
let imageRequestId = 0;

// --------------------------------------------------
// Status and basic helpers
// --------------------------------------------------

function setStatus(message, isError = false) {
    statusBox.textContent = message;
    statusBox.classList.toggle("error", isError);
}

function normalizeText(value) {
    return String(value ?? "").trim();
}

function normalizeKey(value) {
    return normalizeText(value)
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .trim()
        .replace(/\s+/g, " ");
}

function escapeSearchText(value) {
    return normalizeText(value).replace(/\s+/g, " ");
}

async function readJson(response) {
    const responseText = await response.text();

    try {
        return JSON.parse(responseText);
    } catch {
        const status = response.status || "unknown";

        if (response.status === 404) {
            throw new Error(
                "The running backend does not have this API route. Stop it and restart backend/app.py, then refresh the page."
            );
        }

        if (response.status >= 500) {
            throw new Error(
                `The backend returned HTTP ${status} without JSON. Check the Flask terminal for the error details.`
            );
        }

        const contentType = response.headers.get("content-type") || "an unknown content type";
        throw new Error(
            `Expected JSON from the backend but received ${contentType} (HTTP ${status}). Check that the current Flask API is running.`
        );
    }
}

const vehicleSuggestionCache = new Map();
const vehicleSuggestionRequests = new Map();
const vehicleRecordCache = new Map();
const vehicleRecordRequests = new Map();

function keepRecentCacheValue(cache, key, value, maxEntries = 256) {
    if (cache.size >= maxEntries && !cache.has(key)) {
        cache.delete(cache.keys().next().value);
    }
    cache.set(key, value);
}

async function getVehicleSuggestions(query, limit = 10) {
    const cacheKey = `${normalizeKey(query)}|${limit}`;
    if (vehicleSuggestionCache.has(cacheKey)) return vehicleSuggestionCache.get(cacheKey);
    if (vehicleSuggestionRequests.has(cacheKey)) return vehicleSuggestionRequests.get(cacheKey);

    const request = (async () => {
        const response = await fetch(
            `${SUGGESTIONS_URL}?q=${encodeURIComponent(query)}&limit=${limit}`
        );
        const data = await readJson(response);
        if (!response.ok) throw new Error(data.error || "Could not load vehicle suggestions.");
        const suggestions = Array.isArray(data.suggestions)
            ? data.suggestions
            : Array.isArray(data.matches) ? data.matches : [];
        keepRecentCacheValue(vehicleSuggestionCache, cacheKey, suggestions);
        return suggestions;
    })();

    vehicleSuggestionRequests.set(cacheKey, request);
    try {
        return await request;
    } finally {
        if (vehicleSuggestionRequests.get(cacheKey) === request) {
            vehicleSuggestionRequests.delete(cacheKey);
        }
    }
}

let visionModelReady = false;
let visionWarmupPromise = null;
const optimizedPhotoUploads = new WeakMap();

function warmVisionModel() {
    if (visionModelReady) return Promise.resolve(true);
    if (visionWarmupPromise) return visionWarmupPromise;

    visionWarmupPromise = (async () => {
        const response = await fetch(`${API_BASE}/api/vision/warmup`, { method: "POST" });
        const data = await readJson(response);
        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not prepare the local image model.");
        }
        visionModelReady = true;
        return true;
    })().catch(error => {
        visionWarmupPromise = null;
        throw error;
    });

    return visionWarmupPromise;
}

function prepareVehiclePhotoForUpload(file) {
    if (optimizedPhotoUploads.has(file)) return optimizedPhotoUploads.get(file);
    const preparation = (async () => {
        if (file.size <= 512 * 1024 || typeof createImageBitmap !== "function") return file;
        let bitmap;
        try {
            bitmap = await createImageBitmap(file);
            const longestSide = Math.max(bitmap.width, bitmap.height);
            if (longestSide <= 1024 && file.size <= 1024 * 1024) return file;

            const scale = Math.min(1, 1024 / longestSide);
            const canvas = document.createElement("canvas");
            canvas.width = Math.max(1, Math.round(bitmap.width * scale));
            canvas.height = Math.max(1, Math.round(bitmap.height * scale));
            const context = canvas.getContext("2d");
            if (!context) return file;
            context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            const compressed = await new Promise(resolve =>
                canvas.toBlob(resolve, "image/jpeg", 0.82)
            );
            if (!compressed) return file;
            if (longestSide <= 1024 && compressed.size >= file.size) return file;
            const safeName = file.name.replace(/\.[^.]+$/, "") || "vehicle-photo";
            return new File([compressed], `${safeName}.jpg`, { type: "image/jpeg" });
        } catch {
            return file;
        } finally {
            bitmap?.close?.();
        }
    })();
    optimizedPhotoUploads.set(file, preparation);
    return preparation;
}

// --------------------------------------------------
// Autocomplete dropdown
// --------------------------------------------------

function closeSuggestions() {
    suggestionList.classList.add("hidden");
    suggestionList.replaceChildren();
    carInput.setAttribute("aria-expanded", "false");
    carInput.removeAttribute("aria-activedescendant");

    activeSuggestionIndex = -1;
}

function getSuggestionName(item) {
    if (typeof item === "string") {
        return item;
    }

    return normalizeText(
        item.display_name ||
        item.name ||
        item.label ||
        [
            item.make || item.Make || item.manufacturer,
            item.model || item.Model,
            item.variant || item.Trim_Variant || item.trim,
            item.year || item.Year
        ].filter(Boolean).join(" ")
    );
}

function normalizeSuggestions(items) {
    const seen = new Set();
    const normalized = [];

    for (const item of items || []) {
        const name = getSuggestionName(item);

        if (!name) continue;

        const key = normalizeKey(name);
        if (seen.has(key)) continue;

        seen.add(key);

        normalized.push({
            name,
            original: typeof item === "object" && item !== null
                ? item
                : { display_name: name }
        });
    }

    return normalized;
}

function setActiveSuggestion(index) {
    const buttons = suggestionList.querySelectorAll(
        ".suggestion-option"
    );

    if (!buttons.length) return;

    activeSuggestionIndex = Math.max(
        -1,
        Math.min(index, buttons.length - 1)
    );

    buttons.forEach((button, i) => {
        const active = i === activeSuggestionIndex;

        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
    });

    if (activeSuggestionIndex >= 0) {
        const activeButton = buttons[activeSuggestionIndex];

        carInput.setAttribute(
            "aria-activedescendant",
            activeButton.id
        );

        activeButton.scrollIntoView({ block: "nearest" });
    } else {
        carInput.removeAttribute("aria-activedescendant");
    }
}

function renderSuggestions(items, query) {
    suggestionList.replaceChildren();
    suggestionItems = normalizeSuggestions(items);
    activeSuggestionIndex = -1;

    if (!suggestionItems.length) {
        const empty = document.createElement("div");
        empty.className = "suggestion-empty";
        empty.textContent =
            `No matching vehicles for "${query}". Try a shorter model name.`;

        suggestionList.appendChild(empty);
        suggestionList.classList.remove("hidden");
        carInput.setAttribute("aria-expanded", "true");
        return;
    }

    suggestionItems.forEach((item, index) => {
        const button = document.createElement("button");

        button.type = "button";
        button.id = `vehicle-suggestion-${index}`;
        button.className = "suggestion-option";
        button.setAttribute("role", "option");
        button.setAttribute("aria-selected", "false");

        const icon = document.createElement("span");
        icon.className = "suggestion-option-icon";
        icon.textContent = "🚘";
        icon.setAttribute("aria-hidden", "true");

        const content = document.createElement("span");
        content.className = "suggestion-option-content";

        const name = document.createElement("span");
        name.className = "suggestion-option-name";
        name.textContent = item.name;

        const meta = document.createElement("span");
        meta.className = "suggestion-option-meta";

        const record = item.original;
        const year = record.year ?? record.Year;
        const variant = record.variant ?? record.Trim_Variant;
        const details = [variant, year]
            .filter(value =>
                value !== undefined &&
                value !== null &&
                String(value).trim() !== ""
            )
            .map(String);

        meta.textContent = details.length
            ? details.join(" · ")
            : "View vehicle details";

        content.append(name, meta);
        button.append(icon, content);

        // mousedown prevents the input blur from closing the menu first.
        button.addEventListener("mousedown", event => {
            event.preventDefault();
        });

        button.addEventListener("click", () => {
            selectSuggestion(item);
        });

        suggestionList.appendChild(button);
    });

    const footer = document.createElement("div");
    footer.className = "suggestion-footer";
    footer.textContent =
        `${suggestionItems.length} matching vehicle(s)`;

    suggestionList.appendChild(footer);

    suggestionList.classList.remove("hidden");
    carInput.setAttribute("aria-expanded", "true");
}

function selectSuggestion(item) {
    const query = item.name;

    carInput.value = query;
    closeSuggestions();
    searchCar(query);
}

async function fetchSuggestions(query, requestId) {
    try {
        const items = await getVehicleSuggestions(query, 10);

        // Ignore an older request if the user has continued typing.
        if (requestId !== latestSuggestionRequest) return;
        renderSuggestions(items, query);
    } catch (error) {
        if (requestId !== latestSuggestionRequest) return;

        console.error("Autocomplete error:", error);

        // Keep the UI usable if the suggestions endpoint is not ready.
        closeSuggestions();
    }
}

carInput.addEventListener("input", () => {
    clearTimeout(debounceTimer);

    const query = carInput.value.trim();
    const requestId = ++latestSuggestionRequest;

    if (query.length < 2) {
        closeSuggestions();
        return;
    }

    debounceTimer = setTimeout(() => {
        fetchSuggestions(query, requestId);
    }, 250);
});

carInput.addEventListener("keydown", event => {
    const isOpen = !suggestionList.classList.contains("hidden");
    const buttons = suggestionList.querySelectorAll(
        ".suggestion-option"
    );

    if (event.key === "ArrowDown" && isOpen && buttons.length) {
        event.preventDefault();
        setActiveSuggestion(activeSuggestionIndex + 1);
    } else if (
        event.key === "ArrowUp" &&
        isOpen &&
        buttons.length
    ) {
        event.preventDefault();
        setActiveSuggestion(
            activeSuggestionIndex <= 0
                ? buttons.length - 1
                : activeSuggestionIndex - 1
        );
    } else if (
        event.key === "Enter" &&
        isOpen &&
        activeSuggestionIndex >= 0 &&
        suggestionItems[activeSuggestionIndex]
    ) {
        event.preventDefault();
        selectSuggestion(suggestionItems[activeSuggestionIndex]);
    } else if (event.key === "Escape") {
        closeSuggestions();
    }
});

document.addEventListener("click", event => {
    if (
        !searchForm.contains(event.target) &&
        !suggestionList.contains(event.target)
    ) {
        closeSuggestions();
    }
});

// --------------------------------------------------
// Search query preparation
// --------------------------------------------------

function getSearchCandidates(query) {
    const cleaned = escapeSearchText(query);
    const candidates = [cleaned];

    // The dataset search may not include the year in its search text.
    const withoutYear = cleaned
        .replace(/\b(?:19|20)\d{2}\b/g, "")
        .replace(/\s+/g, " ")
        .trim();

    if (withoutYear) candidates.push(withoutYear);

    // Try removing common request words.
    const withoutFiller = withoutYear
        .replace(
            /\b(specifications?|specs|details|information|features|price|car|vehicle)\b/gi,
            ""
        )
        .replace(/\s+/g, " ")
        .trim();

    if (withoutFiller) candidates.push(withoutFiller);

    // If the full make/model/variant does not match, try the base model.
    // This helps find the record, but the selected variant may need a
    // dedicated variant-aware backend endpoint.
    const withoutVariant = withoutFiller
        .replace(/\b(?:Z[2468]L?|Z8L)\b/gi, "")
        .replace(/\s+/g, " ")
        .trim();

    if (withoutVariant) candidates.push(withoutVariant);

    return [...new Set(candidates.filter(Boolean))];
}

// --------------------------------------------------
// Fetch a car record from Flask
// --------------------------------------------------

async function fetchCar(query) {
    const cacheKey = normalizeKey(query);
    if (vehicleRecordCache.has(cacheKey)) return vehicleRecordCache.get(cacheKey);
    if (vehicleRecordRequests.has(cacheKey)) return vehicleRecordRequests.get(cacheKey);

    const request = (async () => {
        const response = await fetch(
            `${API_URL}?q=${encodeURIComponent(query)}`
        );
        const data = await readJson(response);

        if (!response.ok) {
            const error = new Error(
                data.error || `Unable to find "${query}".`
            );
            error.status = response.status;
            error.matches = data.matches || data.car?.matches || [];
            throw error;
        }

        const car = data.car;
        if (!car || typeof car !== "object") {
            throw new Error("The API returned an invalid vehicle record.");
        }
        if (car.ambiguous) {
            const error = new Error(
                "More than one vehicle matched. Choose a specific suggestion."
            );
            error.status = 409;
            error.matches = car.matches || [];
            throw error;
        }

        keepRecentCacheValue(vehicleRecordCache, cacheKey, car);
        return car;
    })();

    vehicleRecordRequests.set(cacheKey, request);
    try {
        return await request;
    } finally {
        if (vehicleRecordRequests.get(cacheKey) === request) {
            vehicleRecordRequests.delete(cacheKey);
        }
    }
}

async function searchCar(query) {
    const requestedQuery = escapeSearchText(query);

    if (!requestedQuery) {
        setStatus("Please enter a car model name.", true);
        return;
    }

    clearTimeout(debounceTimer);
    closeSuggestions();

    searchButton.disabled = true;
    results.classList.add("hidden");
    setStatus("Searching the AutoNex vehicle database...");

    try {
        const candidates = getSearchCandidates(requestedQuery);
        let car = null;
        let lastError = null;

        for (const candidate of candidates) {
            try {
                car = await fetchCar(candidate);
                break;
            } catch (error) {
                lastError = error;

                // Only try a less specific query if this record wasn't found.
                if (error.status !== 404) {
                    throw error;
                }
            }
        }

        if (!car) {
            throw lastError || new Error("No matching vehicle was found.");
        }

        renderCar(car);

        setStatus(
            `Found ${car.name || car.model || requestedQuery}. Specifications loaded.`
        );

        results.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    } catch (error) {
        console.error("Vehicle search error:", error);

        let message = error.message;

        if (error.matches?.length) {
            message += " Available matches: " +
                error.matches.slice(0, 8).join(", ");
        }

        if (
            error.name === "TypeError" ||
            error.message.includes("Failed to fetch")
        ) {
            message =
                "Cannot connect to Flask. Start backend/app.py and " +
                "check that it is running on http://127.0.0.1:5000.";
        }

        setStatus(message, true);
    } finally {
        searchButton.disabled = false;
    }
}

// --------------------------------------------------
// Specifications
// --------------------------------------------------


function renderSpecifications(specifications, car = {}) {
    specGrid.replaceChildren();

    // Fields that should never appear as specification cards.
    const ignoredFields = new Set([
        "car image url",
        "image url",
        "image source url",
        "image source page",
        "source url",
        "source page",
        "source type",
        "description",
        "name",
        "ambiguous",
        "matches"
    ]);

    function shouldDisplayField(key, value) {
        const normalizedKey = normalizeKey(key);

        return (
            !ignoredFields.has(normalizedKey) &&
            value !== null &&
            value !== undefined &&
            typeof value !== "object" &&
            String(value).trim() !== ""
        );
    }

    // Handle specifications already prepared by the backend.
    let fields = Array.isArray(specifications)
        ? specifications.filter(spec => {
            const label =
                spec.label ??
                spec.key ??
                spec.name ??
                spec.field ??
                "";

            return !ignoredFields.has(normalizeKey(label));
        })
        : [];

    // Fallback for APIs that return raw dataset columns.
    if (!fields.length) {
        fields = Object.entries(car)
            .filter(([key, value]) =>
                shouldDisplayField(key, value)
            )
            .map(([label, value]) => ({
                label,
                value
            }));
    }

    // Display a helpful message if nothing remains.
    if (!fields.length) {
        const message = document.createElement("p");

        message.textContent =
            "No specifications are available for this vehicle record.";

        specGrid.appendChild(message);
        return;
    }

    // Render the remaining specification cards.
    fields.forEach(spec => {
        const labelText = normalizeText(
            spec.label ??
            spec.key ??
            spec.name ??
            spec.field ??
            "Specification"
        );

        // Extra protection against image URL fields.
        if (ignoredFields.has(normalizeKey(labelText))) {
            return;
        }

        const specValue =
            spec.value ??
            spec.val ??
            "";

        if (
            specValue === null ||
            specValue === undefined ||
            typeof specValue === "object" ||
            String(specValue).trim() === ""
        ) {
            return;
        }

        const item = document.createElement("div");
        item.className = "spec-item";

        const label = document.createElement("div");
        label.className = "spec-label";
        label.textContent = labelText.replace(/_/g, " ");

        const value = document.createElement("div");
        value.className = "spec-value";
        value.textContent = normalizeText(specValue);

        item.append(label, value);
        specGrid.appendChild(item);
    });
}


// --------------------------------------------------
// Image preview and Wikimedia Commons attribution
// --------------------------------------------------

function resetImage(message = "Searching for a vehicle image...") {
    carImage.onload = null;
    carImage.onerror = null;
    carImage.removeAttribute("src");
    carImage.hidden = true;

    imagePlaceholder.hidden = false;
    imagePlaceholder.querySelector("p").textContent = message;
    imageAttribution.replaceChildren();
}

function addAttributionLink(label, value) {
    if (!value) return;

    try {
        const url = new URL(value);

        if (!["https:", "http:"].includes(url.protocol)) return;

        const link = document.createElement("a");
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = label;

        imageAttribution.appendChild(link);
    } catch {
        // Do not display invalid URLs as clickable links.
    }
}


function displayImage(imageData, currentRequestId, onImageError = null) {
    if (currentRequestId !== imageRequestId) return;

    const imageUrl = normalizeText(
        imageData.image_url ||
        imageData.thumburl ||
        imageData.url
    );

    if (!imageUrl) {
        if (onImageError) {
            onImageError();
        } else {
            resetImage("No suitable vehicle image was found.");
        }
        return;
    }

    let parsedUrl;

    try {
        parsedUrl = new URL(imageUrl);

        if (parsedUrl.protocol !== "https:") {
            throw new Error("Image URL must use HTTPS.");
        }
    } catch {
        if (onImageError) {
            onImageError();
        } else {
            resetImage("The image URL is invalid or unsupported.");
        }
        return;
    }

    resetImage("Loading vehicle image...");

    carImage.onload = () => {
        if (currentRequestId !== imageRequestId) return;

        carImage.hidden = false;
        imagePlaceholder.hidden = true;
    };

    carImage.onerror = () => {
        if (currentRequestId !== imageRequestId) return;

        carImage.onload = null;
        carImage.onerror = null;
        carImage.removeAttribute("src");
        carImage.hidden = true;

        if (onImageError) {
            onImageError();
        } else {
            imagePlaceholder.hidden = false;
            imagePlaceholder.querySelector("p").textContent =
                "This image could not be loaded.";
        }
    };

    carImage.src = parsedUrl.href;

    addAttributionLink(
        "Image source",
        imageData.source_url ||
        imageData.image_source_page ||
        imageData.descriptionurl
    );

    const artist = normalizeText(imageData.artist);
    const license = normalizeText(
        imageData.license || imageData.image_license
    );

    if (artist && artist !== "See source page") {
        const artistText = document.createElement("span");
        artistText.textContent = ` · Creator: ${artist}`;
        imageAttribution.appendChild(artistText);
    }

    if (license) {
        const licenseText = document.createElement("span");
        licenseText.textContent = ` · License: ${license}`;
        imageAttribution.appendChild(licenseText);
    }
}



async function loadCarImage(car) {
    const currentRequestId = ++imageRequestId;

    resetImage("Searching for a vehicle image...");

    const searchName = normalizeText(
        [
            car.manufacturer || car.Make,
            car.model || car.Model
        ].filter(Boolean).join(" ")
    ) || normalizeText(car.name);


    // Prefer local images so they load without an online image search.
    const localCarImages = {
        "tata nexon": "tata_nexon.jpg",
        "tata punch": "tata_punch.jpg",
        "tata harrier": "tata_harrier.jpg",
        "tata safari": "tata_safari.jpg",
        "mahindra scorpio n": "mahindra_scorpio_n.jpg",
        "mahindra thar": "mahindra_thar.jpg",
        "mahindra thar roxx": "mahindra_thar_roxx.jpg",
        "mahindra xuv700": "mahindra_xuv700.jpg",
        "mahindra bolero": "mahindra_bolero.jpg",
        "maruti suzuki swift": "maruti_swift.jpg",
        "maruti suzuki baleno": "maruti_baleno.jpg",
        "maruti suzuki brezza": "maruti_brezza.jpg",
        "maruti suzuki fronx": "maruti_fronx.jpg",
        "maruti suzuki dzire": "maruti_dzire.jpg",
        "hyundai creta": "hyundai_creta.jpg",
        "hyundai venue": "hyundai_venue.jpg",
        "hyundai i20": "hyundai_i20.jpg",
        "kia seltos": "kia_seltos.jpg",
        "toyota innova hycross": "toyota_innova_hycross.jpg",
        "honda city": "honda_city.jpg"
    };

    const localFilename = localCarImages[normalizeKey(searchName)];

    if (localFilename) {
        resetImage("Loading local vehicle image...");

        carImage.onload = () => {
            if (currentRequestId !== imageRequestId) return;

            carImage.hidden = false;
            imagePlaceholder.hidden = true;
            imageAttribution.textContent = "Local AutoNex image";
        };

        carImage.onerror = () => {
            if (currentRequestId !== imageRequestId) return;

            carImage.onload = null;
            carImage.onerror = null;
            loadAlternativeImage();
        };

        carImage.src =
            `${API_BASE}/api/local-car-image/` +
            encodeURIComponent(localFilename);

        return;
    }


    // Search for an alternative image through Flask.
    async function loadAlternativeImage() {
        if (currentRequestId !== imageRequestId) return;

        if (!searchName) {
            resetImage("Vehicle name unavailable for image search.");
            return;
        }

        resetImage("Searching for an alternative vehicle image...");

        try {
            const response = await fetch(
                `${IMAGE_URL}?q=${encodeURIComponent(searchName)}`
            );

            const data = await readJson(response);

            if (currentRequestId !== imageRequestId) return;

            if (!response.ok) {
                resetImage(
                    data.message || "No suitable vehicle image was found."
                );
                return;
            }

            // Do not retry the same failed URL.
            const alternativeUrl = normalizeText(
                data.image_url || data.thumburl || data.url
            );

            const failedDatasetUrl = normalizeText(
                car.image_url || car.Car_Image_URL
            );

            if (
                alternativeUrl &&
                failedDatasetUrl &&
                alternativeUrl === failedDatasetUrl
            ) {
                resetImage("The available vehicle image could not be loaded.");
                return;
            }

            displayImage(data, currentRequestId);
        } catch (error) {
            if (currentRequestId !== imageRequestId) return;

            console.error("Alternative image search error:", error);

            resetImage(
                "Image search failed. Check the Flask backend and image source."
            );
        }
    }

    // Prefer an image provided by the dataset.
    const datasetImage = normalizeText(
        car.image_url || car.Car_Image_URL
    );

    if (datasetImage) {
        displayImage(
            {
                image_url: datasetImage,
                source_url: car.image_source_page || car.image_source_url,
                license: car.image_license,
                artist: car.image_artist
            },
            currentRequestId,
            loadAlternativeImage
        );

        return;
    }

    // No dataset image: search for one directly.
    await loadAlternativeImage();
}


// --------------------------------------------------
// Render the selected vehicle
// --------------------------------------------------

function renderCar(car) {
    carName.textContent = normalizeText(
        car.name || car.model || car.Model
    ) || "Unknown vehicle";

    manufacturerLabel.textContent = normalizeText(
        car.manufacturer || car.Make
    ) || "VEHICLE PROFILE";

    carDescription.textContent = normalizeText(
        car.description
    ) || "Vehicle information retrieved from the AutoNex dataset.";

    sourceText.textContent = normalizeText(
        car.source_type || car.source || "AutoNex vehicle dataset"
    );

    sourceLink.hidden = true;
    sourceLink.removeAttribute("href");

    const sourceUrl = normalizeText(
        car.source_url || car.source_page
    );

    if (sourceUrl) {
        try {
            const url = new URL(sourceUrl);

            if (["https:", "http:"].includes(url.protocol)) {
                sourceLink.href = url.href;
                sourceLink.hidden = false;
            }
        } catch {
            sourceLink.hidden = true;
        }
    }

    renderSpecifications(car.specifications, car);
    results.classList.remove("hidden");

    // Image retrieval does not block the vehicle details.
    loadCarImage(car);
}

// --------------------------------------------------
// Form and quick-search actions
// --------------------------------------------------

searchForm.addEventListener("submit", event => {
    event.preventDefault();
    searchCar(carInput.value);
});

document.querySelectorAll(".suggestion[data-car]").forEach(button => {
    button.addEventListener("click", () => {
        const query = button.dataset.car;

        carInput.value = query;
        closeSuggestions();
        searchCar(query);
    });
});

// Reset autocomplete when the page loses focus.
window.addEventListener("blur", () => {
    clearTimeout(debounceTimer);
});

// Initial state
closeSuggestions();


/* =====================================
   AUTONEX AI COMPARISON FRONTEND
===================================== */

const compareForm = document.getElementById("compareForm");
const compareCar1Input = document.getElementById("compareCar1");
const compareCar2Input = document.getElementById("compareCar2");
const compareQuestion = document.getElementById("compareQuestion");
const compareButton = document.getElementById("compareButton");
const compareStatus = document.getElementById("compareStatus");
const compareResults = document.getElementById("compareResults");
const compareVehicleOptions = document.getElementById("compareVehicleOptions");

function setCompareStatus(message, isError = false) {
    compareStatus.textContent = message;
    compareStatus.classList.toggle("error", isError);
}

function renderComparisonSpecs(containerId, specifications) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.warn(`Comparison specifications container not found: ${containerId}`);
        return;
    }

    container.replaceChildren();

    // Flask may return specifications as a dictionary OR as a list of
    // objects such as [{label: "Price", value: "₹13.55 lakh"}].
    // Object.entries(array) would incorrectly display 0, 1, 2... as labels,
    // and String(object) would display [object Object]. Normalize both shapes.
    const labelKeys = [
        "label", "key", "name", "field", "attribute", "feature",
        "specification", "display_name", "title"
    ];
    const valueKeys = [
        "value", "val", "display_value", "formatted_value", "result",
        "text", "spec_value"
    ];

    function scalarText(value) {
        if (value === null || value === undefined || value === "") return "";
        if (["string", "number", "boolean"].includes(typeof value)) {
            return String(value).trim();
        }
        if (Array.isArray(value)) {
            return value.map(scalarText).filter(Boolean).join(", ");
        }
        return "";
    }

    function objectToEntry(item, fallbackLabel = "") {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
            return [fallbackLabel, scalarText(item)];
        }

        let label = "";
        let value;
        for (const key of labelKeys) {
            if (item[key] !== undefined && item[key] !== null && String(item[key]).trim()) {
                label = String(item[key]).trim();
                break;
            }
        }
        for (const key of valueKeys) {
            if (item[key] !== undefined && item[key] !== null) {
                value = item[key];
                break;
            }
        }

        // Handle nested values like {specification: {label: ..., value: ...}}.
        if (value && typeof value === "object" && !Array.isArray(value)) {
            const nested = objectToEntry(value, label);
            if (!label) label = nested[0];
            value = nested[1];
        }

        // If this is a plain object map, use its scalar properties rather
        // than converting the whole object into a string.
        if (!label && value === undefined) {
            const usable = Object.entries(item)
                .filter(([key, val]) => !labelKeys.includes(key) && scalarText(val));
            if (usable.length === 1) {
                label = usable[0][0];
                value = usable[0][1];
            } else if (usable.length > 1) {
                return [fallbackLabel || "", usable.map(([key, val]) => `${key}: ${scalarText(val)}`).join(" · ")];
            }
        }

        return [label || fallbackLabel || "Specification", scalarText(value)];
    }

    let entries = [];
    if (Array.isArray(specifications)) {
        entries = specifications.map((item, index) =>
            objectToEntry(item, `Specification ${index + 1}`)
        );
    } else if (specifications && typeof specifications === "object") {
        entries = Object.entries(specifications).map(([label, value]) => {
            if (value && typeof value === "object") {
                return objectToEntry(value, label);
            }
            return [label, scalarText(value)];
        });
    }

    entries = entries
        .map(([label, value]) => [String(label || "Specification").replace(/_/g, " ").trim(), scalarText(value)])
        .filter(([label, value]) => label && value && value !== "[object Object]")
        .slice(0, 15);

    if (!entries.length) {
        const message = document.createElement("p");
        message.textContent = "No specifications available for this vehicle.";
        container.appendChild(message);
        return;
    }

    entries.forEach(([label, value]) => {
        const row = document.createElement("div");
        row.className = "comparison-spec-row";

        const labelElement = document.createElement("span");
        labelElement.className = "comparison-spec-label";
        labelElement.textContent = label;

        const valueElement = document.createElement("strong");
        valueElement.className = "comparison-spec-value";
        valueElement.textContent = value;

        row.append(labelElement, valueElement);
        container.appendChild(row);
    });
}

function renderComparisonBasis(basisItems, firstName, secondName) {
    const container = document.getElementById("compareBasisCards");
    container.replaceChildren();

    if (!Array.isArray(basisItems) || basisItems.length === 0) {
        const message = document.createElement("p");
        message.className = "comparison-basis-empty";
        message.textContent = "A result for the selected priority is unavailable for these vehicle records.";
        container.appendChild(message);
        return;
    }

    const verdictLabels = {
        car1: `Priority winner · ${firstName}`,
        car2: `Priority winner · ${secondName}`,
        partial_car1: `Limited data · ${firstName} leads`,
        partial_car2: `Limited data · ${secondName} leads`,
        partial_tie: "Limited data · no clear winner",
        tie: "No winner · tied figures",
        mixed: "No single winner · split result",
        incomplete: "No clear winner · missing data",
        incomparable: "No winner · figures differ in kind",
        shared: "Shared capability · no winner",
        preference: "No single winner · depends on preference",
    };

    basisItems.forEach(item => {
        const card = document.createElement("article");
        card.className = "comparison-basis-card";

        const heading = document.createElement("div");
        heading.className = "comparison-basis-heading";

        const title = document.createElement("h5");
        title.textContent = item.label || "Comparison area";

        const verdict = document.createElement("span");
        verdict.className = "comparison-basis-verdict";
        const verdictCode = item.result || "incomplete";
        verdict.textContent = verdictLabels[verdictCode] || "Review the listed details";
        if (verdictCode === "car1" || verdictCode === "partial_car1") {
            verdict.classList.add("vehicle-one-edge");
        } else if (verdictCode === "car2" || verdictCode === "partial_car2") {
            verdict.classList.add("vehicle-two-edge");
        } else if (verdictCode === "tie" || verdictCode === "shared") {
            verdict.classList.add("basis-tie");
        }

        heading.append(title, verdict);

        const values = document.createElement("div");
        values.className = "comparison-basis-values";
        [[firstName, item.car1_value], [secondName, item.car2_value]].forEach(([name, value]) => {
            const valuePanel = document.createElement("div");
            valuePanel.className = "comparison-basis-value";

            const vehicle = document.createElement("span");
            vehicle.textContent = name;

            const detail = document.createElement("strong");
            detail.textContent = value || "Not available";

            valuePanel.append(vehicle, detail);
            values.appendChild(valuePanel);
        });

        card.append(heading, values);
        container.appendChild(card);
    });
}

let compareSuggestionTimer = null;
let compareSuggestionRequestId = 0;

async function loadCompareSuggestions(query, requestId) {
    try {
        const suggestions = await getVehicleSuggestions(query, 10);
        if (requestId !== compareSuggestionRequestId) return;

        compareVehicleOptions.replaceChildren();
        suggestions.forEach(item => {
            const option = document.createElement("option");
            option.value = item.label || item.name || "";
            if (option.value) compareVehicleOptions.appendChild(option);
        });
    } catch {
        // The text inputs remain usable if autocomplete is unavailable.
    }
}

[compareCar1Input, compareCar2Input].forEach(input => {
    input.addEventListener("input", () => {
        clearTimeout(compareSuggestionTimer);
        const query = input.value.trim();
        const requestId = ++compareSuggestionRequestId;
        if (query.length < 2) {
            compareVehicleOptions.replaceChildren();
            return;
        }
        compareSuggestionTimer = setTimeout(
            () => loadCompareSuggestions(query, requestId),
            220
        );
    });
});

compareForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const car1 = compareCar1Input.value.trim();
    const car2 = compareCar2Input.value.trim();
    const priority = compareQuestion.value;

    if (!car1 || !car2) {
        setCompareStatus("Enter both car names.", true);
        return;
    }

    if (car1.toLowerCase() === car2.toLowerCase()) {
        setCompareStatus(
            "Choose two different vehicles to compare.",
            true
        );
        return;
    }

    compareButton.disabled = true;
    compareResults.classList.add("hidden");
    setCompareStatus("Retrieving vehicle data and generating analysis...");

    try {
        const response = await fetch(`${API_BASE}/api/compare`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                car1,
                car2,
                priority
            })
        });

        const data = await readJson(response);

        if (!response.ok || !data.success) {
            throw new Error(
                data.error || "Unable to compare these vehicles."
            );
        }

        document.getElementById("compareName1").textContent =
            data.car1.name;

        document.getElementById("compareName2").textContent =
            data.car2.name;

        renderComparisonSpecs(
            "compareSpecs1",
            data.car1.specifications
        );

        renderComparisonSpecs(
            "compareSpecs2",
            data.car2.specifications
        );

        const priorityResult = data.priority_result || data.comparison_basis?.[0];
        if (!priorityResult) {
            throw new Error("The backend did not return a result for the selected priority.");
        }

        document.getElementById("compareWinner").textContent =
            priorityResult.headline || "No clear winner could be determined.";
        document.getElementById("compareWinner").className =
            `comparison-winner ${["car1", "car2"].includes(priorityResult.result) ? "has-winner" : "no-winner"}`;

        document.getElementById("compareAnalysis").textContent =
            priorityResult.summary || data.analysis || "No explanation was returned.";

        document.getElementById("compareAnalysisSource").textContent =
            `Rule-based result · ${priorityResult.priority_label || "selected priority"} · listed vehicle data`;

        document.getElementById("comparisonBasesTitle").textContent =
            priorityResult.label || "Selected priority details";

        renderComparisonBasis(
            [priorityResult],
            data.car1.name,
            data.car2.name
        );

        document.getElementById("compareDisclaimer").textContent =
            [data.model, data.notice].filter(Boolean).join(". ") ||
            "Specifications are based on the AutoNex vehicle dataset.";

        compareResults.classList.remove("hidden");

        setCompareStatus(
            `Priority comparison ready for ${(priorityResult.priority_label || "your selection").toLowerCase()}.`
        );
    } catch (error) {
        console.error("AutoNex comparison error:", error);

        setCompareStatus(
            error.message || "Unable to connect to AutoNex AI.",
            true
        );
    } finally {
        compareButton.disabled = false;
    }
});


/* =====================================
   VEHICLE PHOTO SUGGESTION FLOW
===================================== */

const MAX_PHOTO_BYTES = 8 * 1024 * 1024;
let photoPreviewUrl = null;
let singlePhotoWarmupId = 0;

function setRecognitionStatus(message, isError = false) {
    recognitionStatus.textContent = message;
    recognitionStatus.classList.toggle("error", isError);
}

function clearRecognitionMatches() {
    recognitionVehicleSelect.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Choose a suggested make/model";
    recognitionVehicleSelect.appendChild(placeholder);
    recognitionResult.classList.add("hidden");
    loadRecognizedVehicleButton.disabled = true;
}

function releasePhotoPreview() {
    if (photoPreviewUrl) {
        URL.revokeObjectURL(photoPreviewUrl);
        photoPreviewUrl = null;
    }
    vehiclePhotoPreview.removeAttribute("src");
    vehiclePhotoPreview.hidden = true;
}

function clearVehiclePhoto() {
    singlePhotoWarmupId += 1;
    vehiclePhotoInput.value = "";
    releasePhotoPreview();
    uploadPreviewPanel.classList.add("hidden");
    document.getElementById("replacePhotoButton").disabled = true;
    document.getElementById("removePhotoButton").disabled = true;
    recognizeVehicleButton.disabled = true;
    clearRecognitionMatches();
    setRecognitionStatus("Choose a photo to start. The local model prepares as soon as a photo is selected.");
}

function updateConfirmMatchButton() {
    loadRecognizedVehicleButton.disabled = !recognitionVehicleSelect.value.trim();
}

vehiclePhotoInput.addEventListener("change", () => {
    const file = vehiclePhotoInput.files?.[0];
    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
        clearVehiclePhoto();
        setRecognitionStatus("Use a JPEG, PNG, or WebP image.", true);
        return;
    }
    if (file.size > MAX_PHOTO_BYTES) {
        clearVehiclePhoto();
        setRecognitionStatus("Choose an image smaller than 8 MB.", true);
        return;
    }

    releasePhotoPreview();
    clearRecognitionMatches();
    photoPreviewUrl = URL.createObjectURL(file);
    vehiclePhotoPreview.src = photoPreviewUrl;
    vehiclePhotoPreview.hidden = false;
    uploadPreviewPanel.classList.remove("hidden");
    document.getElementById("replacePhotoButton").disabled = false;
    document.getElementById("removePhotoButton").disabled = false;
    const warmupId = ++singlePhotoWarmupId;
    recognizeVehicleButton.disabled = true;
    setRecognitionStatus(`${file.name} is ready. Preparing the local model and vehicle labels while you review the photo…`);
    void prepareVehiclePhotoForUpload(file);
    warmVisionModel().then(() => {
        if (warmupId !== singlePhotoWarmupId || vehiclePhotoInput.files?.[0] !== file) return;
        recognizeVehicleButton.disabled = false;
        setRecognitionStatus(`${file.name} is ready. The local model is prepared; your photo is processed in memory and not saved.`);
    }).catch(error => {
        if (warmupId !== singlePhotoWarmupId) return;
        recognizeVehicleButton.disabled = !vehiclePhotoInput.files?.[0];
        setRecognitionStatus(error.message || "Could not prepare the local image model.", true);
    });
});

document.getElementById("replacePhotoButton").addEventListener("click", () => {
    vehiclePhotoInput.click();
});

document.getElementById("removePhotoButton").addEventListener("click", clearVehiclePhoto);

recognizeVehicleButton.addEventListener("click", async () => {
    const file = vehiclePhotoInput.files?.[0];
    if (!file) {
        setRecognitionStatus("Choose a photo before asking for a match.", true);
        return;
    }

    recognizeVehicleButton.disabled = true;
    clearRecognitionMatches();
    setRecognitionStatus("Preparing the local model and matching the photo against vehicle labels…");

    try {
        const [, uploadFile] = await Promise.all([
            warmVisionModel(),
            prepareVehiclePhotoForUpload(file),
        ]);
        const formData = new FormData();
        formData.append("image", uploadFile, uploadFile.name);
        const response = await fetch(`${API_BASE}/api/recognize`, {
            method: "POST",
            body: formData,
        });
        const data = await readJson(response);
        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not suggest a vehicle from this photo.");
        }

        const predictions = Array.isArray(data.predictions) ? data.predictions : [];
        const vehicleMatches = predictions.filter(item => item.type === "vehicle");
        const categoryMatches = predictions.filter(item => item.type === "category");
        vehicleMatches.forEach(item => {
            const option = document.createElement("option");
            option.value = item.name;
            option.textContent = `${item.name} · ${(item.score * 100).toFixed(1)}% relative score`;
            recognitionVehicleSelect.appendChild(option);
        });

        if (data.suggested_vehicle) {
            recognitionVehicleSelect.value = data.suggested_vehicle;
        }

        document.getElementById("recognitionConfidence").textContent = data.suggested_vehicle
            ? `Top dataset make/model suggestion: ${data.suggested_vehicle} · ${(Number(data.confidence || 0) * 100).toFixed(1)}% relative score. This score is not calibrated certainty.`
            : "No make/model suggestion was confident enough. Search for the vehicle by name above.";
        document.getElementById("recognitionCategory").textContent = categoryMatches.length
            ? `Possible vehicle category: ${categoryMatches[0].name} · approximate only.`
            : "No category suggestion is available.";

        recognitionResult.classList.remove("hidden");
        updateConfirmMatchButton();
        setRecognitionStatus(data.uncertain
            ? "The image match is uncertain. Choose a listed match or search by name above."
            : "Choose the suggested vehicle to load its dataset specifications.");
    } catch (error) {
        setRecognitionStatus(error.message || "Image recognition failed. You can search for the car by name.", true);
    } finally {
        recognizeVehicleButton.disabled = !vehiclePhotoInput.files?.[0];
    }
});

recognitionVehicleSelect.addEventListener("change", updateConfirmMatchButton);

loadRecognizedVehicleButton.addEventListener("click", () => {
    const query = recognitionVehicleSelect.value.trim();
    if (!query) {
        setRecognitionStatus("Choose a suggested make/model before loading dataset specifications.", true);
        return;
    }

    carInput.value = query;
    searchForm.requestSubmit();
});

window.addEventListener("beforeunload", releasePhotoPreview);


/* =====================================
   THREE-PHOTO SPECIFICATION COMPARISON
===================================== */

const threePhotoCompareButton = document.getElementById("threePhotoCompareButton");
const threePhotoCompareStatus = document.getElementById("threePhotoCompareStatus");
const threePhotoCompareResults = document.getElementById("threePhotoCompareResults");
const recognizeThreePhotosButton = document.getElementById("recognizeThreePhotosButton");
const threePhotoBatchStatus = document.getElementById("threePhotoBatchStatus");
let threePhotoBatchBusy = false;
const threePhotoSlotStates = [...document.querySelectorAll(".three-photo-slot")].map(element => ({
    element,
    file: null,
    previewUrl: null,
    car: null,
    busy: false,
    generation: 0,
    variantRequest: 0,
}));

function setThreePhotoStatus(slot, message, isError = false) {
    const status = slot.element.querySelector(".three-photo-status");
    status.textContent = message;
    status.classList.toggle("error", isError);
}

function updateThreePhotoCompareButton() {
    threePhotoCompareButton.disabled = !threePhotoSlotStates.every(slot => slot.car);
}

function updateThreePhotoBatchButton() {
    recognizeThreePhotosButton.disabled = threePhotoBatchBusy ||
        threePhotoSlotStates.some(slot => slot.busy) ||
        !threePhotoSlotStates.every(slot => slot.file);
}

function resetThreePhotoMatch(slot) {
    slot.variantRequest += 1;
    slot.car = null;
    slot.element.querySelector(".three-photo-match").classList.add("hidden");
    const modelSelect = slot.element.querySelector(".three-photo-model");
    const variantSelect = slot.element.querySelector(".three-photo-variant");
    modelSelect.replaceChildren(new Option("Upload and recognize a photo first", ""));
    variantSelect.replaceChildren(new Option("Choose a suggested model first", ""));
    variantSelect.disabled = true;
    slot.element.querySelector(".three-photo-confirm").disabled = true;
    updateThreePhotoCompareButton();
    threePhotoCompareResults.classList.add("hidden");
    threePhotoCompareStatus.textContent = "Add three photos, then confirm a dataset vehicle for each one.";
}

function setThreePhotoBusy(slot, busy) {
    slot.busy = busy;
    slot.element.querySelector(".three-photo-recognize").disabled = busy || !slot.file;
    slot.element.querySelector(".three-photo-remove").disabled = busy || !slot.file;
    const modelSelect = slot.element.querySelector(".three-photo-model");
    const variantSelect = slot.element.querySelector(".three-photo-variant");
    modelSelect.disabled = busy;
    variantSelect.disabled = busy || ![...variantSelect.options].some(option => option.value);
    slot.element.querySelector(".three-photo-confirm").disabled = busy || !variantSelect.value;
    updateThreePhotoBatchButton();
}

function clearThreePhoto(slot) {
    slot.generation += 1;
    slot.variantRequest += 1;
    slot.busy = false;
    slot.file = null;
    slot.car = null;
    if (slot.previewUrl) URL.revokeObjectURL(slot.previewUrl);
    slot.previewUrl = null;
    const input = slot.element.querySelector(".three-photo-input");
    const preview = slot.element.querySelector(".three-photo-preview");
    input.value = "";
    preview.removeAttribute("src");
    preview.hidden = true;
    slot.element.querySelector(".three-photo-prompt").hidden = false;
    slot.element.querySelector(".three-photo-dropzone").classList.remove("has-photo", "is-dragging");
    slot.element.querySelector(".three-photo-remove").disabled = true;
    slot.element.querySelector(".three-photo-recognize").disabled = true;
    resetThreePhotoMatch(slot);
    setThreePhotoStatus(slot, "JPEG, PNG, or WebP · up to 8 MB");
    threePhotoCompareStatus.textContent = "Add three photos, then confirm a dataset vehicle for each one.";
    updateThreePhotoBatchButton();
}

function setThreePhotoFile(slot, file) {
    if (!file) return;
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
        setThreePhotoStatus(slot, "Use a JPEG, PNG, or WebP image.", true);
        return;
    }
    if (file.size > MAX_PHOTO_BYTES) {
        setThreePhotoStatus(slot, "Choose an image smaller than 8 MB.", true);
        return;
    }

    clearThreePhoto(slot);
    slot.file = file;
    slot.previewUrl = URL.createObjectURL(file);
    const preview = slot.element.querySelector(".three-photo-preview");
    preview.src = slot.previewUrl;
    preview.hidden = false;
    slot.element.querySelector(".three-photo-prompt").hidden = true;
    slot.element.querySelector(".three-photo-dropzone").classList.add("has-photo");
    slot.element.querySelector(".three-photo-remove").disabled = false;
    const recognizeButton = slot.element.querySelector(".three-photo-recognize");
    const generation = slot.generation;
    recognizeButton.disabled = true;
    setThreePhotoStatus(slot, `${file.name} is ready. Preparing the local model and vehicle labels while you review the photos…`);
    updateThreePhotoBatchButton();
    void prepareVehiclePhotoForUpload(file);
    warmVisionModel().then(() => {
        if (generation !== slot.generation || slot.file !== file) return;
        recognizeButton.disabled = false;
        setThreePhotoStatus(slot, `${file.name} is ready. The local model is prepared; this photo is processed in memory and not saved.`);
    }).catch(error => {
        if (generation !== slot.generation) return;
        recognizeButton.disabled = false;
        setThreePhotoStatus(slot, error.message || "Could not prepare the local image model.", true);
    });
}

async function loadThreePhotoVariants(slot) {
    const modelSelect = slot.element.querySelector(".three-photo-model");
    const variantSelect = slot.element.querySelector(".three-photo-variant");
    const modelName = modelSelect.value.trim();
    const requestId = ++slot.variantRequest;
    slot.car = null;
    variantSelect.replaceChildren(new Option("Loading dataset variants…", ""));
    variantSelect.disabled = true;
    slot.element.querySelector(".three-photo-confirm").disabled = true;
    updateThreePhotoCompareButton();
    threePhotoCompareResults.classList.add("hidden");
    threePhotoCompareStatus.textContent = "Confirm the selected dataset variant for all three cars to compare their specifications.";
    if (!modelName) {
        variantSelect.replaceChildren(new Option("Choose a suggested model first", ""));
        return;
    }

    try {
        const suggestions = await getVehicleSuggestions(modelName, 10);
        if (requestId !== slot.variantRequest) return;
        variantSelect.replaceChildren(new Option(
            suggestions.length ? "Choose a variant and year" : "No matching dataset records",
            ""
        ));
        suggestions.forEach(item => {
            if (!item.label) return;
            const option = new Option(item.label, item.label);
            variantSelect.appendChild(option);
        });
        variantSelect.disabled = !suggestions.length || slot.busy;
        setThreePhotoStatus(slot, suggestions.length
            ? "Choose the exact variant and year shown in the dataset."
            : "No dataset variants were found for that model. Try another suggestion.", !suggestions.length);
    } catch (error) {
        if (requestId !== slot.variantRequest) return;
        variantSelect.replaceChildren(new Option("Dataset variants unavailable", ""));
        setThreePhotoStatus(slot, error.message || "Could not load dataset variants.", true);
    }
}

async function showThreePhotoPrediction(slot, data) {
    const matches = (Array.isArray(data?.predictions) ? data.predictions : [])
        .filter(item => item.type === "vehicle");
    if (!matches.length) {
        throw new Error("No make/model suggestions were returned for one photo. Try a clearer picture.");
    }

    const modelSelect = slot.element.querySelector(".three-photo-model");
    modelSelect.replaceChildren(new Option("Choose a suggested make/model", ""));
    matches.forEach(item => {
        modelSelect.appendChild(new Option(
            `${item.name} · ${(Number(item.score || 0) * 100).toFixed(1)}% relative score`,
            item.name,
        ));
    });
    if (!data.uncertain && data.suggested_vehicle && matches.some(item => item.name === data.suggested_vehicle)) {
        modelSelect.value = data.suggested_vehicle;
    }

    slot.element.querySelector(".three-photo-match").classList.remove("hidden");
    setThreePhotoStatus(slot, data.uncertain
        ? "The image suggestion is uncertain. Choose a make/model before loading dataset variants."
        : "Choose the dataset variant and year for this model.");
    if (modelSelect.value) await loadThreePhotoVariants(slot);
}

threePhotoSlotStates.forEach(slot => {
    const input = slot.element.querySelector(".three-photo-input");
    const dropzone = slot.element.querySelector(".three-photo-dropzone");
    const removeButton = slot.element.querySelector(".three-photo-remove");
    const recognizeButton = slot.element.querySelector(".three-photo-recognize");
    const modelSelect = slot.element.querySelector(".three-photo-model");
    const variantSelect = slot.element.querySelector(".three-photo-variant");
    const confirmButton = slot.element.querySelector(".three-photo-confirm");

    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            input.click();
        }
    });
    input.addEventListener("change", () => setThreePhotoFile(slot, input.files?.[0]));
    removeButton.addEventListener("click", () => clearThreePhoto(slot));
    modelSelect.addEventListener("change", () => loadThreePhotoVariants(slot));
    variantSelect.addEventListener("change", () => {
        slot.car = null;
        confirmButton.disabled = !variantSelect.value || slot.busy;
        updateThreePhotoCompareButton();
        threePhotoCompareResults.classList.add("hidden");
        threePhotoCompareStatus.textContent = "Confirm the selected dataset variant for all three cars to compare their specifications.";
    });

    ["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
    }));
    ["dragleave", "dragend"].forEach(type => dropzone.addEventListener(type, event => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
    }));
    dropzone.addEventListener("drop", event => {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
        setThreePhotoFile(slot, event.dataTransfer?.files?.[0]);
    });

    recognizeButton.addEventListener("click", async () => {
        if (!slot.file) return;
        const generation = ++slot.generation;
        setThreePhotoBusy(slot, true);
        resetThreePhotoMatch(slot);
        setThreePhotoBusy(slot, true);
        setThreePhotoStatus(slot, "Checking this photo against the vehicle models…");
        try {
            const [, uploadFile] = await Promise.all([
                warmVisionModel(),
                prepareVehiclePhotoForUpload(slot.file),
            ]);
            const body = new FormData();
            body.append("image", uploadFile, uploadFile.name);
            const response = await fetch(`${API_BASE}/api/recognize`, { method: "POST", body });
            const data = await readJson(response);
            if (generation !== slot.generation) return;
            if (!response.ok || !data.success) throw new Error(data.error || "Could not suggest a vehicle from this photo.");
            await showThreePhotoPrediction(slot, data);
        } catch (error) {
            if (generation === slot.generation) setThreePhotoStatus(slot, error.message || "Image recognition failed.", true);
        } finally {
            if (generation === slot.generation) setThreePhotoBusy(slot, false);
        }
    });

    confirmButton.addEventListener("click", async () => {
        const query = variantSelect.value.trim();
        if (!query) return;
        const generation = slot.generation;
        setThreePhotoBusy(slot, true);
        setThreePhotoStatus(slot, "Loading the selected dataset specifications…");
        try {
            const car = normalizeThreePhotoCar(await fetchCar(query));
            if (generation !== slot.generation) return;
            if (car.name === "Unknown vehicle") {
                throw new Error("The vehicle record did not include a make or model. Choose another dataset result.");
            }
            if (!car.specifications.length) {
                throw new Error("The selected dataset record did not include readable specifications.");
            }
            slot.car = car;
            const variant = car.variant && car.variant !== "Not available" ? ` · ${car.variant}` : "";
            const year = car.year && car.year !== "Not available" ? ` · ${car.year}` : "";
            setThreePhotoStatus(slot, `Specifications loaded: ${car.name}${variant}${year}.`);
            updateThreePhotoCompareButton();
        } catch (error) {
            if (generation === slot.generation) setThreePhotoStatus(slot, error.message || "Could not load that dataset record.", true);
        } finally {
            if (generation === slot.generation) setThreePhotoBusy(slot, false);
        }
    });
});

recognizeThreePhotosButton.addEventListener("click", async () => {
    if (!threePhotoSlotStates.every(slot => slot.file) || threePhotoBatchBusy) return;

    const files = threePhotoSlotStates.map(slot => slot.file);
    const generations = threePhotoSlotStates.map(slot => slot.generation);
    threePhotoBatchBusy = true;
    updateThreePhotoBatchButton();
    threePhotoBatchStatus.classList.remove("error");
    threePhotoBatchStatus.textContent = "Preparing the local model, then matching all three photos in one batch…";
    threePhotoSlotStates.forEach(slot => setThreePhotoBusy(slot, true));

    try {
        const [, uploadFiles] = await Promise.all([
            warmVisionModel(),
            Promise.all(files.map(prepareVehiclePhotoForUpload)),
        ]);
        const body = new FormData();
        uploadFiles.forEach(file => body.append("images", file, file.name));
        const response = await fetch(`${API_BASE}/api/recognize/batch`, {
            method: "POST",
            body,
        });
        const data = await readJson(response);
        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not match the three photos.");
        }
        if (!Array.isArray(data.results) || data.results.length !== 3) {
            throw new Error("The backend did not return three photo matches.");
        }
        const unchanged = threePhotoSlotStates.every((slot, index) =>
            slot.generation === generations[index] && slot.file === files[index]
        );
        if (!unchanged) {
            throw new Error("A photo changed during matching. Click this button again to match the current photos.");
        }

        await Promise.all(threePhotoSlotStates.map((slot, index) =>
            showThreePhotoPrediction(slot, data.results[index])
        ));
        threePhotoBatchStatus.textContent = "All three photos were matched in one model request. Review each suggestion and select its dataset variant.";
    } catch (error) {
        threePhotoBatchStatus.textContent = error.message || "Three-photo matching failed.";
        threePhotoBatchStatus.classList.add("error");
    } finally {
        threePhotoSlotStates.forEach(slot => setThreePhotoBusy(slot, false));
        threePhotoBatchBusy = false;
        updateThreePhotoBatchButton();
    }
});

function normalizeThreePhotoCar(record) {
    const source = record && typeof record === "object" ? record : {};
    const fields = new Map(Object.entries(source).map(([key, value]) => [normalizeKey(key), value]));
    const read = (...names) => {
        for (const name of names) {
            const value = fields.get(normalizeKey(name));
            if (value !== null && value !== undefined && typeof value !== "object") {
                const text = String(value).trim();
                if (text && !["nan", "null", "none"].includes(text.toLowerCase())) return text;
            }
        }
        return "";
    };
    const cleanSpecValue = value => {
        if (value === null || value === undefined || typeof value === "object") return "";
        const text = String(value).trim();
        return !text || ["nan", "null", "none", "not available"].includes(text.toLowerCase())
            ? ""
            : text;
    };

    const manufacturer = read("manufacturer", "make");
    const model = read("model");
    const existingName = read("name");
    const name = existingName || [manufacturer, model].filter(Boolean).join(" ") || "Unknown vehicle";
    const variant = read("variant", "trim variant") || "Not available";
    let year = read("year") || "Not available";
    if (/^\d{4}\.0+$/.test(year)) year = String(Number(year));

    const suppliedSpecs = fields.get("specifications");
    let specifications = [];
    if (Array.isArray(suppliedSpecs)) {
        specifications = suppliedSpecs.map(item => ({
            label: item?.label ?? item?.key ?? item?.name ?? item?.field ?? "",
            value: cleanSpecValue(item?.value ?? item?.val ?? item?.display_value ?? ""),
        })).filter(item => item.label && item.value);
    } else if (suppliedSpecs && typeof suppliedSpecs === "object") {
        specifications = Object.entries(suppliedSpecs)
            .map(([label, value]) => ({ label, value: cleanSpecValue(value) }))
            .filter(item => item.value);
    } else {
        const formattedPrice = read("Formatted_Price");
        const rawPrice = read("Ex_Showroom_Price_INR");
        const datasetFields = [
            ["Body style", read("Body_Style")],
            ["Doors", read("Doors")],
            ["Seating capacity", read("Seating_Capacity")],
            ["Drivetrain", read("Drivetrain")],
            ["Transmission", read("Transmission")],
            ["Fuel type", read("Fuel_Type")],
            ["Engine displacement", read("Displacement")],
            ["Cylinders", read("Cylinders")],
            ["Power", read("Power_BHP")],
            ["City mileage", read("Mileage_City")],
            ["Highway mileage", read("Mileage_Highway")],
            [formattedPrice ? "Ex-showroom price" : "Ex-showroom price (INR)", formattedPrice || rawPrice],
        ];
        specifications = [
            ["Manufacturer", manufacturer], ["Model", model], ["Year", year], ["Variant", variant],
            ...datasetFields,
        ].map(([label, value]) => ({ label, value: cleanSpecValue(value) }))
            .filter(item => item.value);
    }

    return { ...source, name, manufacturer, model, variant, year, specifications };
}

function threePhotoSpecMap(car) {
    const values = new Map();
    (Array.isArray(car.specifications) ? car.specifications : []).forEach(item => {
        const label = normalizeKey(item.label ?? item.key ?? item.name ?? "");
        const value = normalizeText(item.value ?? item.val ?? "");
        if (label && value && value.toLowerCase() !== "not available") values.set(label, value);
    });
    return values;
}

function threePhotoValue(car, specs, keys) {
    if (keys.includes("year") && car.year && car.year !== "Not available") return car.year;
    if (keys.includes("variant") && car.variant && car.variant !== "Not available") return car.variant;
    for (const key of keys) {
        if (specs.has(key)) return specs.get(key);
    }
    if (keys.includes("price")) {
        for (const [key, value] of specs) {
            if (key.startsWith("ex showroom price")) return value;
        }
    }
    return "Not listed";
}

function renderThreePhotoSpecs() {
    const groups = [
        { title: "Model and price", rows: [
            ["Vehicle", ["model"]], ["Year", ["year"]], ["Variant", ["variant"]], ["Ex-showroom price", ["price"]],
        ] },
        { title: "Engine and economy", rows: [
            ["Fuel type", ["fuel type"]], ["Engine displacement", ["engine displacement"]], ["Cylinders", ["cylinders"]],
            ["Power", ["power"]], ["City mileage", ["city mileage"]], ["Highway mileage", ["highway mileage"]],
        ] },
        { title: "Space and driving", rows: [
            ["Body style", ["body style"]], ["Doors", ["doors"]], ["Seating capacity", ["seating capacity"]],
            ["Drivetrain", ["drivetrain"]], ["Transmission", ["transmission"]],
        ] },
    ];
    const table = document.createElement("table");
    table.className = "three-photo-spec-table";
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    const specHeading = document.createElement("th");
    specHeading.scope = "col";
    specHeading.textContent = "Specification";
    headerRow.appendChild(specHeading);
    threePhotoSlotStates.forEach((slot, index) => {
        const th = document.createElement("th");
        th.scope = "col";
        const car = slot.car;
        th.textContent = `${index + 1}. ${car.name}${car.variant && car.variant !== "Not available" ? ` · ${car.variant}` : ""}${car.year && car.year !== "Not available" ? ` · ${car.year}` : ""}`;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    const specMaps = threePhotoSlotStates.map(slot => threePhotoSpecMap(slot.car));
    groups.forEach(group => {
        const groupRow = document.createElement("tr");
        groupRow.className = "three-photo-spec-group";
        const groupCell = document.createElement("th");
        groupCell.colSpan = 4;
        groupCell.scope = "rowgroup";
        groupCell.textContent = group.title;
        groupRow.appendChild(groupCell);
        tbody.appendChild(groupRow);
        group.rows.forEach(([label, keys]) => {
            const row = document.createElement("tr");
            const heading = document.createElement("th");
            heading.scope = "row";
            heading.textContent = label;
            row.appendChild(heading);
            threePhotoSlotStates.forEach((slot, index) => {
                const cell = document.createElement("td");
                cell.textContent = label === "Vehicle"
                    ? slot.car.name
                    : threePhotoValue(slot.car, specMaps[index], keys);
                row.appendChild(cell);
            });
            tbody.appendChild(row);
        });
    });
    table.appendChild(tbody);
    threePhotoCompareResults.replaceChildren(table);
    threePhotoCompareResults.classList.remove("hidden");
    threePhotoCompareStatus.textContent = "Specifications from the three selected dataset records are shown side by side.";
}

threePhotoCompareButton.addEventListener("click", () => {
    if (!threePhotoSlotStates.every(slot => slot.car)) return;
    renderThreePhotoSpecs();
});

window.addEventListener("beforeunload", () => {
    threePhotoSlotStates.forEach(slot => {
        if (slot.previewUrl) URL.revokeObjectURL(slot.previewUrl);
    });
});
