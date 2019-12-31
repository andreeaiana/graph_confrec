# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle
import requests
import pandas as pd
from tqdm import tqdm
from urllib import response
from bs4 import BeautifulSoup


class Scraper:

    def __init__(self):
        self.base_url = "https://scholar.google.com/"

        dir_persistent = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..", "..", "data", "interim", "H5Index")
        if not os.path.exists(dir_persistent):
            os.makedirs(dir_persistent)
        self.path_persistent = os.path.join(
                dir_persistent, "h5_index_rankings.pkl")

    def scrape(self):
        if not self._load_rankings():
            print("H5-Index rankings not scraped yet. Scraping now...\n")
            categories = self._scrape_categories()
            subcategories = self._scrape_subcategories(categories)
            self.rankings = self._scrape_rankings(subcategories)
            print("Finished.\n")
            self._save_rankings()
        else:
            print("H5-Index rankings already persistent.\n")

        # Show some statistics
        print("Statistics")
        print("\tNumber categories: {}.".format(len(
                self.rankings.category.unique())))
        print("\tNumber subcategories: {}.".format(len(
                self.rankings.subcategory.unique())))
        print("\tNumber rankings: {}.\n".format(len(
                self.rankings.h5_index)))
        df = self.rankings.groupby("category")["subcategory"].agg(
                set).reset_index()
        df["len"] = df.subcategory.apply(lambda x: len(x))
        min_len = df["len"].min()
        max_len = df["len"].max()
        print("\tMean number of subcategories: {}.".format(df["len"].mean()))
        print("\tMin number of subcategories is {} for category {}.".format(
                min_len, df[df["len"] == min_len]))
        print("\tMax number of subcategories is {} for category {}.".format(
                max_len, df[df["len"] == max_len]))

    def _scrape_categories(self):
        print("Scraping categories...")
        url = self.base_url + "/citations?view_op=top_venues&hl=en"
        try:
            data = requests.get(url)
        except Exception as e:
            print(str(e))

        if data.status_code != 200:
            raise ConnectionError(
                    "Failed to open url: {} (status code: {}).".format(
                            url, response.status_code))

        soup = BeautifulSoup(data.text, 'lxml')
        categories = list()
        for item in soup.find_all(
                'a', attrs={"class": "gs_md_li",
                            "role": "menuitem", "tabindex": "-1"}):
            category = item.get_text()
            link = item.get("href")
            if len(category.split()) > 1:
                categories.append((category, self.base_url + link))
        categories_df = pd.DataFrame(categories, columns=["category", "url"])
        print("Scraped {} categories.\n".format(len(categories_df)))
        return categories_df

    def _scrape_subcategories(self, categories):
        print("Scraping subcategories...")
        subcategories = list()
        count_categories = len(categories)
        with tqdm(desc="Scraping subcategories: ",
                  total=count_categories) as pbar:
            for idx in range(count_categories):
                url = categories.url.iloc[idx]
                try:
                    data = requests.get(url)
                except Exception as e:
                    print(str(e))

                if data.status_code != 200:
                    raise ConnectionError(
                            "Failed to open url: {} (status code: {}).".format(
                                    url, response.status_code))

                soup = BeautifulSoup(data.text, "lxml")
                for item in soup.find_all(
                        'a', attrs={"class": "gs_md_li", "role": "menuitem",
                                    "tabindex": "-1"}):
                    subcategory = item.get_text()
                    link = item.get("href")
                    if len(subcategory.split()) > 1:
                        subcategories.append((
                                categories.category.iloc[idx],
                                subcategory,
                                self.base_url + link))
                time.sleep(5)
                pbar.update(1)
        subcategories_df = pd.DataFrame(
                subcategories, columns=["category", "subcategory", "url"])
        print("Scraped {} subcategories for {} categories.\n".format(
                len(subcategories_df), count_categories))
        return subcategories_df

    def _scrape_rankings(self, subcategories):
        print("Scraping rankings...")
        category = list()
        subcategory = list()
        publication = list()
        h5_index = list()
        h5_median = list()
        count_subcategories = len(subcategories)
        with tqdm(desc="Scraping rankings: ",
                  total=count_subcategories) as pbar:
            for idx in range(count_subcategories):
                url = subcategories.url.iloc[idx]
                try:
                    data = requests.get(url)
                except Exception as e:
                    print(str(e))
                print(data.status_code)
                if data.status_code != 200:
                    raise ConnectionError(
                            "Failed to open url: {} (status code: {}).".format(
                                    url, response.status_code))

                soup = BeautifulSoup(data.text, "lxml")
                for content in soup.body.find_all(
                        'table', attrs={'class' : 'gsc_mp_table'}):
                    for table in content.find_all('tr'):
                        i = 0
                        for infos in table.find_all("td"):
                            if i == 1:
                                publication.append(infos.get_text())
                            if i == 2:
                                h5_index.append(infos.get_text())
                            if i == 3:
                                h5_median.append(infos.get_text())
                            i += 1

                category.extend([subcategories.category.iloc[idx]] * 20)
                subcategory.extend(
                        [subcategories.subcategory.iloc[idx]] * 20)
                time.sleep(5)
                pbar.update(1)

        rankings = pd.DataFrame({
                "category": category, "subcategory": subcategory,
                "publication": publication, "h5_index": h5_index,
                "h5_median": h5_median})
        print("Finished scraping {} rankings for {} subcategories and {} categories.\n".format(
                len(rankings.h5_index), len(rankings.category.unique()),
                len(rankings.subcategory.unique())))
        return rankings

    def _save_rankings(self):
        print("Saving rankings to disk...")
        with open(self.path_persistent, "wb") as f:
            pickle.dump(self.rankings, f)
        print("Saved.\n")

    def _load_rankings(self):
        if os.path.isfile(self.path_persistent):
            print("Loading H5-index rankings...")
            with open(self.path_persistent, "rb") as f:
                self.rankings = pickle.load(f)
            print("Loaded.\n")
            return True
        return False

    def main():
        print("Starting...\n")
        from H5IndexScraper import Scraper
        scraper = Scraper()
        scraper.scrape()
        print("Finished.")

    if __name__ == "__main__":
        main()
