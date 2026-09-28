import os

from main import main

DEFAULT_OPTIONS_PATH = 'wrds_options_raw.csv'
DEFAULT_STOCK_PATH = 'wrds_stock_raw.csv'

def prompt_for_csv_path(prompt_label, default_path):
    while True:
        raw = input(f"{prompt_label} [{default_path}]: ").strip()
        path = raw or default_path
        if os.path.exists(path):
            return path
        print(f"  File not found: {path!r}. Try again.")

if __name__ == "__main__":
    options_path = prompt_for_csv_path("Options CSV filename", DEFAULT_OPTIONS_PATH)
    stock_path = prompt_for_csv_path("Stock CSV filename", DEFAULT_STOCK_PATH)
    main(options_path, stock_path)
