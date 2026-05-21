import pandas as pd
import numpy as np
from pcf_scraping_functions import apple_func, samsung_func, lenovo_func, dell_func #import scraper functions
from datetime import date



def full_scraper(csv): #takes in previous csv

    if csv is not None: #if a csv was given
        csv = csv #continue
        old_pdfs = set(csv["PDF_Link"]) #grab all previous links
    else: #otherwise if no previous csv
        csv = pd.DataFrame() #create empty df
        old_pdfs = set() #empty set of old pdfs
    
    #pass in old pdf links to functions, except Samsung since they host all products on few pdfs
    dell_df = dell_func(old_pdfs)
    apple_df = apple_func(old_pdfs)
    samsung_df = samsung_func()
    lenovo_df = lenovo_func(old_pdfs)


    #combine results of all scrapers into df
    new_data = pd.concat([dell_df, apple_df, samsung_df, lenovo_df])
    new_data["date_requested"] = date.today() #timestamp for current date

    #combine new data with previous csv
    full_df = pd.concat([csv, new_data], ignore_index=True)
    #drop duplicate products
    full_df.drop_duplicates(subset=['Product_Name', 'Brand'], keep='first', inplace=True) 
    
    #save to csv
    full_df.to_csv("scraper_pcf_info.csv", index=False)
    return full_df

#Running the scraper. Simply run this code. If there is a previous csv, ensure it is named "scraper_pcf_info.csv", or change the name in the function. If there is no previous csv, it will still run.
try:
    df = full_scraper(pd.read_csv("scraper_pcf_info.csv"))
except FileNotFoundError:
    df = full_scraper(csv=None)