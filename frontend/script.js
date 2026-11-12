
const API_BASE = "http://127.0.0.1:5000";
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
    try {
        return await response.json();
    } catch {
        throw new Error("The backend returned an invalid JSON response.");
    }
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
        const response = await fetch(
            `${SUGGESTIONS_URL}?q=${encodeURIComponent(query)}&limit=10`
        );

        const data = await readJson(response);

        // Ignore an older request if the user has continued typing.
        if (requestId !== latestSuggestionRequest) return;

        if (!response.ok) {
            throw new Error(data.error || "Could not load suggestions.");
        }

        const items = data.suggestions || data.matches || [];

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

    return car;
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

        const data = await response.json();

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

        document.getElementById("compareAnalysis").textContent =
            data.analysis || "No analysis was returned.";

        document.getElementById("compareAnalysisSource").textContent =
            data.model_used
                ? "Generated by AutoNex Tiny GPT from the listed vehicle data"
                : "Grounded in the listed vehicle data";

        document.getElementById("compareDisclaimer").textContent =
            [data.model, data.notice].filter(Boolean).join(". ") ||
            "Specifications are based on the AutoNex vehicle dataset.";

        compareResults.classList.remove("hidden");

        setCompareStatus(
            data.model_used
                ? "Comparison generated with AutoNex Tiny GPT."
                : "Comparison generated from the available vehicle data."
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
