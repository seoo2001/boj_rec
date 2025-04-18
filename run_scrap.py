from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from scraper.entity import Base
from scraper.scrap import scrap_user, scrap_problem, scrap_interaction
import argparse


DATABASE_URL = "sqlite:///data/raw/baekjoon.db"
engine = create_engine(DATABASE_URL, echo=True)

Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

scraper_map = {
    "user": scrap_user,
    "problem": scrap_problem,
    "interaction": scrap_interaction
}

def main():
    args = parser.parse_args()
    scrape_type = args.type
    start_page = args.start_page

    if scrape_type in scraper_map:
        print(f"Scraping {scrape_type} data...")
        scraper_map[scrape_type](db, start_page)
        print(f"{scrape_type.capitalize()} data scraped successfully.")
    else:
        print(f"Unknown scrape type: {scrape_type}")
        return
    db.commit()
    db.close()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape data from Solved.ac")
    parser.add_argument(
        "--type",
        "-t",
        choices=["user", "problem", "interaction"],
        required=True,
        help="Type of data to scrape: user, problem, or interaction"
    )
    parser.add_argument(
        "--start_page",
        "-s",
        type=int,
        default=1,
        help="Starting page for user scraping (default: 1)"
    )
    main()