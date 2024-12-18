from typing import Any
from re import sub

from csv import DictReader
from pathlib import Path

from json import dump

def main() -> None:
    raw_data_folder = Path("./raw_data/")

    parsed_q_n_a: list[dict[str, Any]] = []

    for file_path in raw_data_folder.glob("./*"):
        if file_path.is_dir():
            continue

        with open(file_path, "r", newline="", encoding="utf-8") as q_n_a_csv:
            reader = DictReader(q_n_a_csv)

            for raw_q_n_a in reader:
                raw_category = raw_q_n_a.pop("Category").lower().strip()

                category_without_specials = sub(
                    r"\[.*\]",
                    "",
                    raw_category
                )

                if raw_category != category_without_specials:
                    continue

                categories = (
                    "_".join(category.split()) for
                    category in
                    (
                        sub(
                            r'\W+',
                            ' ',
                            category
                        ) for
                        category in
                        category_without_specials.split(", ")
                    )
                )

                cleaned_keys = {
                    key.lower().replace(" ", "_"): value for
                    key, value in
                    raw_q_n_a.items() if 
                    len(value) >= 1 and 
                    all(
                        (word not in key) for 
                        word in
                        ("Time", "Date") 
                    )
                }

                for category in categories:
                    cleaned_keys["category"] = category
                    parsed_q_n_a.append(cleaned_keys)

    with open("parsed.json", "w", encoding="utf-8") as parsed_json_file:
        dump(parsed_q_n_a, parsed_json_file, indent=4)


if __name__ == "__main__":
    main()
