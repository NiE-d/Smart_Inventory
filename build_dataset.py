
from pathlib import Path
import time
import requests
import imagehash
from PIL import Image
from io import BytesIO

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "dataset_raw"

CATEGORIES = {
    "resistor": [
        "axial resistor electronic component",
        "through hole resistor colored bands",
    ],
    "arduino": [
        "Arduino Uno board",
        "Arduino microcontroller board",
    ],
    "breadboard": [
        "solderless breadboard electronics",
        "electronics prototyping breadboard",
    ],
    "led": [
        "through hole LED electronic component",
        "light emitting diode component",
    ],
    "buzzer": [
        "piezo buzzer electronic component",
        "electronic buzzer component",
    ],
}

MAX_IMAGES_PER_CLASS = 200
API_URL = "https://commons.wikimedia.org/w/api.php"

session = requests.Session()
session.headers.update({
    "User-Agent": "SmartInventoryDataset/1.0 (educational project)"
})


def search_images(query, limit=100):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "iiurlwidth": 800,
        "format": "json",
    }

    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    return [
        page["imageinfo"][0]
        for page in pages.values()
        if page.get("imageinfo")
        and page["imageinfo"][0].get("mime", "").startswith("image/")
    ]


def collect_category(category, queries):
    folder = RAW_DIR / category
    folder.mkdir(parents=True, exist_ok=True)

    seen_hashes = []
    for existing in folder.glob("*.jpg"):
        try:
            with Image.open(existing) as im:
                seen_hashes.append(imagehash.phash(im.convert("RGB")))
        except Exception:
            pass

    saved = len(list(folder.glob("*.jpg")))
    print(f"\n--- {category.upper()} ---")

    for query in queries:
        if saved >= MAX_IMAGES_PER_CLASS:
            break

        print("Searching:", query)

        try:
            results = search_images(query)
        except Exception as exc:
            print("Search failed:", exc)
            continue

        for item in results:
            if saved >= MAX_IMAGES_PER_CLASS:
                break

            url = item.get("thumburl") or item.get("url")
            if not url:
                continue

            try:
                response = session.get(url, timeout=25)
                response.raise_for_status()

                with Image.open(BytesIO(response.content)) as im:
                    im = im.convert("RGB")
                    if im.width < 100 or im.height < 100:
                        continue

                    phash = imagehash.phash(im)
                    if any(phash - old < 5 for old in seen_hashes):
                        continue

                    saved += 1
                    destination = folder / f"{category}_{saved:03d}.jpg"
                    im.save(destination, "JPEG", quality=92)
                    seen_hashes.append(phash)

                    print(f"Saved {saved}: {destination.name}")

                time.sleep(0.2)

            except Exception:
                continue

    print(f"Total saved for {category}: {saved}")
    print("Inspect the images manually; search results may be mislabeled.")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for category, queries in CATEGORIES.items():
        collect_category(category, queries)

    print("\nCollection finished.")
    print("Review dataset_raw before retraining.")
    print("Check image licenses before redistribution or publication.")


if __name__ == "__main__":
    main()
