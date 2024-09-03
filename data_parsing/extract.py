from csv import DictReader
from pathlib import Path

def main() -> None:
    raw_data_folder = Path("./1_raw_data/")

    q_and_as = {

    }

    for file_path in raw_data_folder.glob("./*"):
        file_stem = file_path.stem

        if file_stem.endswith("conversations"):
            place_name = file_stem[:-len("_conversations")]
            print(place_name)
            continue
            
        with open(file_path, "r", newline="", encoding="utf-8") as place_q_and_a_file:
            reader = DictReader(place_q_and_a_file)

            for raw_q_and_a in reader:
                print(raw_q_and_a)

if __name__ == "__main__":
    main()