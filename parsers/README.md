# Парсеры

## SoFIFA по годам (`sofifa_by_year.py`)

Предсказание рейтинга **на следующий год после сезона**: матч в сезоне 2016/2017 → таргет = рейтинг SoFIFA за 2017 год.

### Запуск

Из корня проекта:

```bash
python parsers/sofifa_by_year.py
```

Скрипт:
1. Читает `dataset/data_with_dates.csv`, извлекает уникальные сезоны.
2. Для каждого сезона вычисляет `rating_year` (год рейтинга: следующий после сезона).
3. Скачивает список игроков SoFIFA за этот год (страницы `?r=<roster_id>&offset=...`) и сохраняет в `dataset/sofifa_players_by_year/{year}.csv`.
4. Собирает итоговую таблицу `dataset/sofifa_ratings_by_season.csv` (player_name, season_name, rating_year, overall) для обучения.

### Зависимости

- `requests`, `beautifulsoup4` (уже в корневом `requirements.txt`).

Если SoFIFA отдаёт пустую таблицу (контент подгружается по JS), можно:
- использовать браузерный автоматизацию (Selenium) и сохранять HTML в файлы, либо
- библиотеку [soccerdata](https://github.com/probberechts/soccerdata) (`SoFIFA(versions=...)`) и доработать скрипт под неё.

### Roster ID по годам

В скрипте задан словарь `ROSTER_BY_YEAR` (roster ID с сайта SoFIFA для каждого года). При необходимости его можно уточнить по выпадающему списку версий на https://www.sofifa.com/ .
