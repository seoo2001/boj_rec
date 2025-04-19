from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.scraper import parallel_scrap_problem_per_page, parallel_scrap_user_per_page, parallel_crawl_interaction
from src.data.entity import Base
from dotenv import load_dotenv
from sqlalchemy import create_engine
import argparse
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/raw/baekjoon.db")

engine = create_engine(DATABASE_URL, echo=True)
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

scraper_map = {
    "user": parallel_scrap_user_per_page,
    "problem": parallel_scrap_problem_per_page,
    "interaction": parallel_crawl_interaction,
}

def main():
    parser = argparse.ArgumentParser(description="Scrape data from Solved.ac")
    parser.add_argument(
        "--type",
        "-t",
        choices=["user", "problem", "interaction"],
        required=True,
        help="Type of data to scrape: user, problem, or interaction"
    )
    args = parser.parse_args()
    scrape_type = args.type
    if scrape_type in scraper_map:
        print(f"Scraping {scrape_type} data...")
        scraper_map[scrape_type](db)
        print(f"{scrape_type.capitalize()} data scraped successfully.")
    else:
        print(f"Unknown scrape type: {scrape_type}")
        return
    
if __name__ == "__main__":
    main()
    db.close()