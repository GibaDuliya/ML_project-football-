"""
Парсер рейтингов SoFIFA по годам для матчей из data_with_dates.csv.

Логика: матч в сезоне 2016/2017 → предсказываем рейтинг из следующего года (2017).
Таргет: рейтинг SoFIFA за год, следующий за сезоном матча.

Использование:
  python parsers/sofifa_by_year.py

Читает dataset/data_with_dates.csv, определяет уникальные сезоны, для каждого
сезона вычисляет rating_year (год рейтинга: следующий после сезона), скачивает
рейтинги SoFIFA за этот год и сохраняет в dataset/sofifa_players_by_year/{year}.csv.
Итоговая таблица для обучения: dataset/sofifa_ratings_by_season.csv
(player_name, season_name, rating_year, overall).
"""

from __future__ import annotations

import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

# cloudscraper обходит Cloudflare (как в risingBALLER/scripts/scrape_sofifa.py)
try:
    import cloudscraper
    _scraper = cloudscraper.create_scraper()
    USE_CLOUDSCRAPER = True
except ImportError:
    _scraper = requests.Session()
    USE_CLOUDSCRAPER = False

# Ростер SoFIFA по году (r= в URL). Год = следующий после сезона (рейтинг за этот год).
# Источник: dropdown на sofifa.com. Для недостающих лет используется эвристика YY0042.
# SoFIFA покрывает в основном FIFA 98 и новее; для более старых лет данных может не быть.
ROSTER_BY_YEAR = {
    1998: "980042",
    1999: "990042",
    2000: "000042",
    2001: "010042",
    2002: "020042",
    2003: "030042",
    2004: "040042",
    2005: "050042",
    2006: "060042",
    2007: "070042",
    2008: "080042",
    2009: "090042",
    2010: "100042",
    2011: "110042",
    2012: "120042",
    2013: "130042",
    2014: "140042",
    2015: "150042",
    2016: "160042",
    2017: "170042",   # FIFA 17
    2018: "180027",   # FIFA 18
    2019: "190001",
    2020: "200145",
    2021: "210135",
    2022: "220069",
    2023: "230034",   # FIFA 23
    2024: "240016",   # FIFA 24
    2025: "250003",   # FIFA 25
}


def get_roster_id(year: int) -> str | None:
    """Возвращает roster ID для года; если нет в словаре — пробуем эвристику YY0042 (FIFA YY)."""
    if year in ROSTER_BY_YEAR:
        return ROSTER_BY_YEAR[year]
    # Эвристика для лет без явной записи (SoFIFA обычно есть с FIFA 98)
    if 1998 <= year <= 99:
        return f"{year % 100:02d}0042"
    if 2000 <= year <= 2025:
        return f"{year % 100:02d}0042"
    return None

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}
BASE_URL = "https://sofifa.com/"
RATE_LIMIT_SLEEP = 0.6
MAX_RETRIES = 5


def normalize_name(name: str) -> str:
    """Нормализация имени для сопоставления (lowercase, пробелы, дефисы)."""
    if not name or not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[-–—]", " ", s).strip()
    return s


def season_to_rating_year(season_name: str) -> int | None:
    """Сезон матча → год рейтинга (следующий год после сезона).

    Примеры: 2016/2017 -> 2017, 2023/2024 -> 2024, 2018 -> 2019.
    """
    if pd.isna(season_name) or not str(season_name).strip():
        return None
    s = str(season_name).strip()
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
    if s.isdigit():
        return int(s) + 1
    return None


def fetch_page(url: str):
    """Загрузка страницы с повторами; cloudscraper обходит Cloudflare."""
    for attempt in range(MAX_RETRIES):
        try:
            r = _scraper.get(url, headers=REQUEST_HEADERS, timeout=30)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(random.uniform(1, 3))
    return None


def get_available_versions(soup: BeautifulSoup) -> dict[str, str]:
    """Доступные версии FIFA/FC из dropdown на главной (version_key -> r= param)."""
    versions = {}
    select = soup.find("select", {"name": "version"})
    if not select:
        return versions
    for opt in select.find_all("option"):
        value = opt.get("value", "")
        if "r=" in value:
            match = re.search(r"r=([^&]+)", value)
            if match:
                versions[opt.get_text(strip=True)] = match.group(1)
    return versions


def year_to_version_key(year: int) -> str:
    """Год -> ключ в dropdown: FC 24 (2024+), FIFA 17 (до 2024)."""
    yy = str(year % 100).zfill(2)
    return f"FC {yy}" if year >= 2024 else f"FIFA {yy}"


def parse_players_from_html(html: str, base_url: str = BASE_URL) -> list[dict[str, str | float]]:
    """Парсинг таблицы игроков: data-col (oa=overall), data-tippy-content или текст ссылки для имени (как в risingBALLER)."""
    soup = BeautifulSoup(html, "html.parser")
    players = []
    table = soup.find("table")
    if not table:
        return players
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
        name_link = row.find("a", href=re.compile(r"/player/\d+"))
        if not name_link:
            continue
        name = name_link.get("data-tippy-content") or name_link.get("data-tooltip") or name_link.get_text(strip=True)
        if not name:
            continue
        player: dict[str, str | float | None] = {"name": name, "overall": None}
        for td in tds:
            data_col = td.get("data-col")
            if not data_col:
                continue
            text = td.get_text(strip=True)
            if data_col == "oa":
                player["overall"] = int(text) if text.isdigit() else None
                break
        if player["overall"] is None:
            nums = re.findall(r"\d+", " ".join(t.get_text(strip=True) for t in tds[2:10]))
            if len(nums) >= 2:
                player["overall"] = int(nums[1])
        if player["overall"] is not None and 40 <= player["overall"] <= 99:
            players.append({"name": name, "overall": float(player["overall"])})
    return players


def get_next_page_url(soup: BeautifulSoup, base_url: str = BASE_URL) -> str | None:
    """Ссылка на следующую страницу (Next в pagination)."""
    pagination = soup.find("div", class_="pagination")
    if not pagination:
        return None
    for a in pagination.find_all("a"):
        if a.get_text(strip=True) == "Next":
            href = a.get("href")
            if href:
                return urljoin(base_url, href)
    return None


def scrape_sofifa_year(roster_id: str, max_pages: int | None = None) -> list[dict]:
    """Скрапит всех игроков за один ростер. URL и пагинация как в risingBALLER (Next link)."""
    start_url = f"https://sofifa.com/?r={roster_id}&col=oa&sort=desc&offset=0"
    all_players = []
    url = start_url
    page = 1
    pbar = tqdm(desc=f"r={roster_id}", unit=" page", dynamic_ncols=True)
    try:
        while True:
            r = fetch_page(url)
            if not r:
                break
            batch = parse_players_from_html(r.text, BASE_URL)
            if not batch:
                break
            all_players.extend(batch)
            pbar.set_postfix(players=len(all_players))
            pbar.update(1)
            if max_pages is not None and page >= max_pages:
                break
            soup = BeautifulSoup(r.text, "html.parser")
            next_url = get_next_page_url(soup, BASE_URL)
            if not next_url or next_url == url:
                break
            url = next_url
            page += 1
            time.sleep(random.uniform(RATE_LIMIT_SLEEP, RATE_LIMIT_SLEEP + 0.4))
    finally:
        pbar.close()
    return all_players


def fetch_sofifa_year_selenium(roster_id: str, max_pages: int = 50, page_size: int = 60) -> list[dict]:
    """Опционально: скрапит через Selenium (если контент подгружается по JS)."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        return []
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    all_players = []
    try:
        driver = webdriver.Chrome(options=options)
        for page in range(max_pages):
            offset = page * page_size
            url = f"https://www.sofifa.com/players?r={roster_id}&set=true&offset={offset}"
            driver.get(url)
            time.sleep(0.8)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.list tr"))
                )
            except Exception:
                pass
            html = driver.page_source
            batch = parse_players_from_html(html)
            if not batch:
                break
            all_players.extend(batch)
            if len(batch) < page_size:
                break
        driver.quit()
    except Exception:
        pass
    return all_players


def copy_latest_sofifa_csv_to_year(root: Path, year: int) -> bool:
    """Копирует dataset/sofifa_players.csv в dataset/sofifa_players_by_year/{year}.csv (колонки name, overall)."""
    src = root / "dataset" / "sofifa_players.csv"
    if not src.exists():
        return False
    df = pd.read_csv(src)
    if "name" not in df.columns or "overall" not in df.columns:
        return False
    out_dir = root / "dataset" / "sofifa_players_by_year"
    out_dir.mkdir(parents=True, exist_ok=True)
    df[["name", "overall"]].to_csv(out_dir / f"{year}.csv", index=False)
    return True


def get_rating_years_from_matches(csv_path: Path) -> set[int]:
    """Из data_with_dates.csv получает множество rating_year для всех сезонов."""
    df = pd.read_csv(csv_path)
    if "season_name" not in df.columns:
        return set()
    years = set()
    for sn in df["season_name"].dropna().unique():
        y = season_to_rating_year(sn)
        if y is not None:
            years.add(y)
    return years


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Scrape SoFIFA ratings by year (next year after season).")
    ap.add_argument("--max-pages", type=int, default=None, help="Max pages per year (default: all)")
    ap.add_argument("--year", type=int, default=None, help="Only fetch this rating year (default: all)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_with_dates = root / "dataset" / "data_with_dates.csv"
    out_dir = root / "dataset" / "sofifa_players_by_year"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_with_dates.exists():
        print("File not found:", data_with_dates)
        return

    # Roster IDs с сайта (dropdown на главной), иначе fallback на ROSTER_BY_YEAR
    versions_by_key: dict[str, str] = {}
    print("Loading sofifa.com for version list...")
    r_main = fetch_page(BASE_URL)
    if r_main:
        soup_main = BeautifulSoup(r_main.text, "html.parser")
        versions_by_key = get_available_versions(soup_main)
        if versions_by_key:
            print("  Versions on site:", list(versions_by_key.keys())[:15], "...")
    else:
        print("  Main page failed, using ROSTER_BY_YEAR only.")

    rating_years = get_rating_years_from_matches(data_with_dates)
    rating_years = sorted(rating_years)
    if args.year is not None:
        rating_years = [y for y in rating_years if y == args.year]
        print("Rating year (single):", rating_years)
    else:
        print("Rating years from data_with_dates (all):", rating_years)

    for year in rating_years:
        version_key = year_to_version_key(year)
        roster_id = versions_by_key.get(version_key)
        if not roster_id and versions_by_key:
            for k, r in versions_by_key.items():
                if k.strip().startswith(version_key) or version_key in k:
                    roster_id = r
                    break
        roster_id = roster_id or get_roster_id(year)
        if not roster_id:
            print(f"  {year}: no roster ID (SoFIFA has no version for this year), skip.")
            continue
        out_csv = out_dir / f"{year}.csv"
        if out_csv.exists():
            print(f"  {year}: skip (exists). Delete {out_csv} to re-fetch.")
            continue
        print(f"  {year}: fetching {version_key} r={roster_id} ...")
        players = scrape_sofifa_year(roster_id, max_pages=args.max_pages)
        if not players and year == 2025:
            if copy_latest_sofifa_csv_to_year(root, year):
                print(f"  {year}: used dataset/sofifa_players.csv (current snapshot).")
                continue
        if not players:
            players = fetch_sofifa_year_selenium(roster_id)
        if not players:
            print(f"  {year}: no data (install selenium for JS-rendered page).")
            continue
        pd.DataFrame(players).drop_duplicates(subset=["name"], keep="first").to_csv(
            out_csv, index=False
        )
        print(f"  {year}: saved {len(players)} rows -> {out_csv}")

    # Итоговая таблица: для каждого (player_name, season_name) из матчей — overall из rating_year
    matches_df = pd.read_csv(data_with_dates)
    matches_df = matches_df[["player_name", "season_name"]].drop_duplicates()
    matches_df["rating_year"] = matches_df["season_name"].map(season_to_rating_year)
    matches_df = matches_df.dropna(subset=["rating_year"])
    matches_df["rating_year"] = matches_df["rating_year"].astype(int)

    merged_rows = []
    for year in matches_df["rating_year"].unique():
        year_csv = out_dir / f"{year}.csv"
        if not year_csv.exists():
            continue
        rating_df = pd.read_csv(year_csv)
        if "name" not in rating_df.columns or "overall" not in rating_df.columns:
            continue
        rating_df = rating_df.rename(columns={"name": "player_name"})
        rating_df["name_norm"] = rating_df["player_name"].astype(str).map(normalize_name)
        rating_df = rating_df.drop_duplicates(subset=["name_norm"], keep="first")
        subset = matches_df[matches_df["rating_year"] == year][["player_name", "season_name", "rating_year"]].copy()
        subset["name_norm"] = subset["player_name"].astype(str).map(normalize_name)
        subset = subset.merge(
            rating_df[["name_norm", "overall"]],
            on="name_norm",
            how="left",
        ).drop(columns=["name_norm"])
        merged_rows.append(subset)
    if merged_rows:
        merged = pd.concat(merged_rows, ignore_index=True)
        out_merged = root / "dataset" / "sofifa_ratings_by_season.csv"
        merged.to_csv(out_merged, index=False)
        print("Merged table:", out_merged, "rows:", len(merged))
    else:
        print("No data for merged table. Fetch year CSVs first.")


if __name__ == "__main__":
    main()
